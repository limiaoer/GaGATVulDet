from __future__ import print_function
from parser import parameter_parser
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from sklearn.utils import compute_class_weight
from sklearn.metrics import confusion_matrix

tf.compat.v1.set_random_seed(9906)
args = parameter_parser()



class GeGATModel:

    def __init__(self, graph_train, graph_test, y_train, y_test,
                 batch_size=args.batch_size, lr=args.lr, epochs=args.epochs,
                 node_in_features=1000, node_out_features=1000):

        self.graph_train = graph_train
        self.graph_test = graph_test
        self.y_train = y_train
        self.y_test = y_test
        self.batch_size = batch_size
        self.epochs = epochs


        self.node_in_features = node_in_features
        self.node_out_features = node_out_features
        self.edge_in_features = 4
        self.global_in_features = 4


        self._init_data()
        self._init_model(lr)

    def _init_data(self):
        self.class_weight = compute_class_weight(
            'balanced',
            classes=[0, 1],
            y=self.y_train
        )

        self.graph_train_tensor = torch.FloatTensor(self.graph_train)
        self.graph_test_tensor = torch.FloatTensor(self.graph_test)
        self.y_train_tensor = torch.FloatTensor(self.y_train.reshape(-1, 1))
        self.y_test_tensor = torch.FloatTensor(self.y_test.reshape(-1, 1))

        self._generate_auxiliary_features()

    def _generate_auxiliary_features(self):
        batch_size, _, _ = self.graph_train.shape

        self.edge_train = torch.randn(batch_size, 2, self.edge_in_features)  # 假设2条边
        self.edge_test = torch.randn(len(self.graph_test), 2, self.edge_in_features)

        self.global_train = torch.randn(batch_size, self.global_in_features)
        self.global_test = torch.randn(len(self.graph_test), self.global_in_features)

        self._generate_adjacency_matrices()

    def _generate_adjacency_matrices(self):
        num_nodes = 1
        num_edges = 2

        self.adj_nodes = torch.eye(num_nodes)

        self.adj_edges = torch.zeros(num_edges, num_edges)
        for j in range(num_edges):
            if j > 0:
                self.adj_edges[j, j - 1] = 1
            if j < num_edges - 1:
                self.adj_edges[j, j + 1] = 1

    def _init_model(self, lr):
        self.model = TemporalGraphNetwork(
            node_in_features=self.node_in_features,
            edge_in_features=self.edge_in_features,
            global_in_features=self.global_in_features,
            node_out_features=self.node_out_features
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.BCELoss()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def train(self):
        self.model.train()

        dataset = torch.utils.data.TensorDataset(
            self.graph_train_tensor,
            self.y_train_tensor
        )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        for epoch in range(self.epochs):
            total_loss = 0
            for batch_idx, (batch_graph, batch_y) in enumerate(loader):
                batch_edge = self.edge_train[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]
                batch_global = self.global_train[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]

                batch_graph = batch_graph.to(self.device)
                batch_edge = batch_edge.to(self.device)
                batch_global = batch_global.to(self.device)
                batch_y = batch_y.to(self.device)
                adj_nodes = self.adj_nodes.to(self.device)
                adj_edges = self.adj_edges.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(
                    batch_graph,
                    batch_edge,
                    batch_global,
                    adj_nodes,
                    adj_edges
                )


                weights = torch.tensor([
                    self.class_weight[1] if y == 1 else self.class_weight[0]
                    for y in batch_y.cpu().numpy().flatten()
                ], device=self.device)

                loss = self.criterion(outputs, batch_y)
                loss = (loss * weights).mean()

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f'Epoch {epoch + 1}/{self.epochs}, Loss: {total_loss / len(loader):.4f}')

    def test(self):
        self.model.eval()
        with torch.no_grad():
            graph_test = self.graph_test_tensor.to(self.device)
            edge_test = self.edge_test.to(self.device)
            global_test = self.global_test.to(self.device)
            adj_nodes = self.adj_nodes.to(self.device)
            adj_edges = self.adj_edges.to(self.device)

            outputs = self.model(
                graph_test,
                edge_test,
                global_test,
                adj_nodes,
                adj_edges
            )
            predictions = (outputs > 0.5).float().cpu()

            loss = self.criterion(outputs.cpu(), self.y_test_tensor)
            accuracy = (predictions == self.y_test_tensor).float().mean()

            tn, fp, fn, tp = confusion_matrix(
                self.y_test,
                predictions.numpy().flatten()
            ).ravel()

            print("\nTest Results:")
            print(f'Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}')
            print(f"Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
            print(f'False positive rate(FPR): {fp / (fp + tn):.4f}')
            print(f'False negative rate(FNR): {fn / (fn + tp):.4f}')
            recall = tp / (tp + fn)
            print(f'Recall(TPR): {recall:.4f}')
            precision = tp / (tp + fp)
            print(f'Precision: {precision:.4f}')
            print(f'F1 score: {(2 * precision * recall) / (precision + recall):.4f}')


class TemporalGraphNetwork(nn.Module):

    def __init__(self, node_in_features, edge_in_features, global_in_features,
                 node_out_features, num_heads=4):
        super(TemporalGraphNetwork, self).__init__()

        self.graph_attn = TemporalGraphLayer(
            node_in_features=node_in_features,
            edge_in_features=edge_in_features,
            global_in_features=global_in_features,
            node_out_features=node_out_features,
            edge_out_features=4,  # 默认值
            num_heads=num_heads
        )

        self.dense1 = nn.Linear(node_out_features, 200)
        self.dense2 = nn.Linear(200, 100)
        self.output = nn.Linear(100, 1)

        self.dropout = nn.Dropout(0.5)

    def forward(self, V, E, U, adj_nodes, adj_edges):

        V_new, _ = self.graph_attn(V, E, U, adj_nodes, adj_edges)

        x = F.relu(self.dense1(V_new.squeeze(1)))
        x = self.dropout(x)
        x = F.relu(self.dense2(x))
        x = self.dropout(x)
        return torch.sigmoid(self.output(x))


class TemporalGraphLayer(nn.Module):

    def __init__(self, node_in_features, edge_in_features, global_in_features,
                 node_out_features, edge_out_features, num_heads, dropout=0.6, alpha=0.2):
        super(TemporalGraphLayer, self).__init__()
        self.node_in_features = node_in_features
        self.edge_in_features = edge_in_features
        self.global_in_features = global_in_features
        self.node_out_features = node_out_features
        self.edge_out_features = edge_out_features
        self.num_heads = num_heads
        self.alpha = alpha

        # 节点特征变换参数 (对应伪代码中的W^k)
        self.node_W = Parameter(torch.FloatTensor(num_heads, node_in_features, node_out_features))

        # 边特征变换参数 (对应伪代码中的W^k)
        self.edge_W = Parameter(torch.FloatTensor(num_heads, edge_in_features, edge_out_features))

        # 全局特征变换参数 (对应伪代码中的P_0)
        self.global_P0 = Parameter(torch.FloatTensor(num_heads, global_in_features, node_out_features))

        # 直接相连节点变换参数 (对应伪代码中的P_m)
        self.direct_Pm = Parameter(torch.FloatTensor(num_heads, node_in_features, node_out_features))

        # 注意力机制参数 (对应伪代码中的a)
        self.edge_attention_a = Parameter(torch.FloatTensor(num_heads, 2 * edge_out_features, 1))
        self.node_attention_a = Parameter(torch.FloatTensor(num_heads, 2 * node_out_features, 1))

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.activation = nn.ELU()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.node_W.data, gain=1.414)
        nn.init.xavier_uniform_(self.edge_W.data, gain=1.414)
        nn.init.xavier_uniform_(self.global_P0.data, gain=1.414)
        nn.init.xavier_uniform_(self.direct_Pm.data, gain=1.414)
        nn.init.xavier_uniform_(self.edge_attention_a.data, gain=1.414)
        nn.init.xavier_uniform_(self.node_attention_a.data, gain=1.414)

    def forward(self, V, E, U, adj_nodes, adj_edges):
        batch_size, N, _ = V.size()
        M = E.size(1)

        E_transformed = torch.einsum('hij,bmj->bmhi', self.edge_W, E)
        V_transformed = torch.einsum('hij,bnj->bnhi', self.node_W, V)
        V_direct = torch.einsum('hij,bnj->bnhi', self.direct_Pm, V)
        U_transformed = torch.einsum('hij,bj->bhi', self.global_P0, U).unsqueeze(1)

        E_temporal = []
        for j in range(M):
            neighbors = []
            if j > 0:  # j-1
                neighbors.append(j - 1)
            if j < M - 1:  # j+1
                neighbors.append(j + 1)

            if not neighbors:
                E_temporal.append(torch.zeros(batch_size, self.num_heads, self.edge_out_features, device=E.device))
                continue

            e_j = E_transformed[:, j].unsqueeze(2)
            e_n = E_transformed[:, neighbors]

            concat_features = torch.cat([e_j.expand(-1, len(neighbors), -1, -1), e_n], dim=-1)
            e = torch.einsum('hij,bmnhj->bmnh', self.edge_attention_a, concat_features)
            e = self.leakyrelu(e)

            attention = F.softmax(e, dim=1)
            attention = self.dropout(attention)

            aggregated = torch.sum(attention.unsqueeze(-1) * e_n, dim=1)
            E_temporal.append(aggregated)

        E_temporal = torch.stack(E_temporal, dim=1)

        E_direct = []
        for j in range(M):
            m = j % N
            m_neighbors = [m, (m + 1) % N]

            e_j = E_transformed[:, j].unsqueeze(2)
            v_m = V_direct[:, m_neighbors]

            concat_features = torch.cat([e_j.expand(-1, 2, -1, -1), v_m], dim=-1)
            e = torch.einsum('hij,bmnhj->bmnh', self.edge_attention_a, concat_features)
            e = self.leakyrelu(e)

            attention = F.softmax(e, dim=1)
            aggregated = torch.sum(attention.unsqueeze(-1) * v_m, dim=1)
            E_direct.append(aggregated)

        E_direct = torch.stack(E_direct, dim=1)

        E_global = E_transformed + U_transformed

        E_new = self.activation((E_temporal + E_direct + E_global + E_transformed) / 4.0)
        E_new = E_new.mean(dim=2)

        V_transformed = torch.einsum('hij,bnj->bnhi', self.node_W, V)
        E_transformed = torch.einsum('hij,bmj->bmhi', self.edge_W, E_new)

        V_neighbors = []
        for i in range(N):
            neighbors = torch.nonzero(adj_nodes[i]).squeeze(-1)
            if neighbors.numel() == 0:
                V_neighbors.append(torch.zeros(batch_size, self.num_heads, self.node_out_features, device=V.device))
                continue

            v_i = V_transformed[:, i].unsqueeze(2)
            v_n = V_transformed[:, neighbors]

            concat_features = torch.cat([v_i.expand(-1, len(neighbors), -1, -1), v_n], dim=-1)
            e = torch.einsum('hij,bmnhj->bmnh', self.node_attention_a, concat_features)
            e = self.leakyrelu(e)

            attention = F.softmax(e, dim=1)
            aggregated = torch.sum(attention.unsqueeze(-1) * v_n, dim=1)
            V_neighbors.append(aggregated)

        V_neighbors = torch.stack(V_neighbors, dim=1)

        V_edges = []
        for i in range(N):
            connected_edges = [i % M, (i + 1) % M]

            v_i = V_transformed[:, i].unsqueeze(2)
            e_n = E_transformed[:, connected_edges]

            concat_features = torch.cat([v_i.expand(-1, len(connected_edges), -1, -1), e_n], dim=-1)
            e = torch.einsum('hij,bmnhj->bmnh', self.node_attention_a, concat_features)
            e = self.leakyrelu(e)

            attention = F.softmax(e, dim=1)
            aggregated = torch.sum(attention.unsqueeze(-1) * e_n, dim=1)
            V_edges.append(aggregated)

        V_edges = torch.stack(V_edges, dim=1)

        V_global = V_transformed + U_transformed.expand(-1, N, -1, -1)

        V_new = self.activation((V_neighbors + V_edges + V_global + V_transformed) / 4.0)
        V_new = V_new.mean(dim=2)

        return V_new, E_new


class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, num_heads, dropout=0.6, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.alpha = alpha
        self.concat = concat
        self.W = Parameter(torch.FloatTensor(num_heads, in_features, out_features))
        self.a = Parameter(torch.FloatTensor(num_heads, 2 * out_features, 1))
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h, adj):
        batch_size, N, _ = h.size()
        h_transformed = torch.einsum('hij,bnj->bnh', self.W, h)
        h_i = h_transformed.unsqueeze(2).expand(-1, -1, N, -1, -1)
        h_j = h_transformed.unsqueeze(1).expand(-1, N, -1, -1, -1)

        concat_features = torch.cat([h_i, h_j], dim=-1)

        e = torch.einsum('hij,bmnhi->bmnh', self.a, concat_features)
        e = self.leakyrelu(e)

        mask = -1e20 * (1.0 - adj.unsqueeze(-1))
        e = e + mask.unsqueeze(0)

        attention = F.softmax(e, dim=2)
        attention = self.dropout(attention)

        h_prime = torch.einsum('bmnh,bnhi->bmhi', attention, h_transformed)

        if self.concat:
            h_prime = h_prime.reshape(batch_size, N, -1)
        else:
            h_prime = h_prime.mean(dim=2)

        return h_prime



# if __name__ == "__main__":
#     import numpy as np
#
#     graph_train = np.random.rand(100, 1, 256)  # 100样本，每个1节点，256维
#     graph_test = np.random.rand(20, 1, 256)
#     y_train = np.random.randint(0, 2, 100)
#     y_test = np.random.randint(0, 2, 20)
#
#     # 创建模型
#     model = GeGATModel(
#         graph_train=graph_train,
#         graph_test=graph_test,
#         y_train=y_train,
#         y_test=y_test,
#         batch_size=16,
#         lr=0.001,
#         epochs=5
#     )
#
#
#     model.train()
#     model.test()