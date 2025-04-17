import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import compute_class_weight
from sklearn.metrics import confusion_matrix
from parser import parameter_parser


args = parameter_parser()


class GNNModel:
    def __init__(self, graph_train, graph_test, y_train, y_test,
                 batch_size=args.batch_size, lr=args.lr, epochs=args.epochs,
                 hidden_size=1000, propagation_rounds=1000):

        self.graph_train = graph_train
        self.graph_test = graph_test
        self.y_train = y_train
        self.y_test = y_test
        self.batch_size = batch_size
        self.epochs = epochs
        self.hidden_size = hidden_size
        self.propagation_rounds = propagation_rounds
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

        self._generate_adjacency_matrices()

    def _generate_adjacency_matrices(self):
        num_nodes = self.graph_train.shape[1]
        self.adj_matrix = torch.ones(num_nodes, num_nodes)

    def _init_model(self, lr):
        input_size = self.graph_train.shape[-1]

        self.model = GNNCore(
            input_size=input_size,
            hidden_size=self.hidden_size,
            propagation_rounds=self.propagation_rounds
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.BCELoss()

        # GPU支持
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
            for batch_graph, batch_y in loader:
                batch_graph = batch_graph.to(self.device)
                batch_y = batch_y.to(self.device)
                adj_matrix = self.adj_matrix.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_graph, adj_matrix)

                # 应用类别权重
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
            adj_matrix = self.adj_matrix.to(self.device)

            outputs = self.model(graph_test, adj_matrix)
            predictions = (outputs > 0.5).float().cpu()

            # 计算指标
            loss = self.criterion(outputs.cpu(), self.y_test_tensor)
            accuracy = (predictions == self.y_test_tensor).float().mean()

            # 混淆矩阵
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


class GNNCore(nn.Module):
    def __init__(self, input_size, hidden_size, propagation_rounds=2):
        super(GNNCore, self).__init__()
        self.hidden_size = hidden_size
        self.propagation_rounds = propagation_rounds

        # 输入变换
        self.input_transform = nn.Linear(input_size, hidden_size)

        # 边权重 (简化处理，只使用一种边类型)
        self.edge_weights = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size))

        # GRU单元
        self.gru_cell = nn.GRUCell(hidden_size, hidden_size)

        # 输出层
        self.output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input_transform.weight)
        nn.init.xavier_uniform_(self.edge_weights)
        nn.init.zeros_(self.input_transform.bias)

    def forward(self, node_features, adj_matrix):
        batch_size, num_nodes, _ = node_features.size()

        h = self.input_transform(node_features)  # (batch_size, num_nodes, hidden_size)

        for _ in range(self.propagation_rounds):
            messages = torch.matmul(h, self.edge_weights)  # (batch_size, num_nodes, hidden_size)
            aggregated = torch.matmul(adj_matrix.unsqueeze(0), messages)  # (batch_size, num_nodes, hidden_size)


            h = self.gru_cell(
                aggregated.view(-1, self.hidden_size),
                h.view(-1, self.hidden_size)
            ).view(batch_size, num_nodes, self.hidden_size)

        graph_embedding = h.mean(dim=1)

        return self.output(graph_embedding)



# if __name__ == "__main__":
#     import numpy as np
#
#     graph_train = np.random.rand(100, 1, 256)
#     graph_test = np.random.rand(20, 1, 256)
#     y_train = np.random.randint(0, 2, 100)
#     y_test = np.random.randint(0, 2, 20)
#
#     model = GNNModel(
#         graph_train=graph_train,
#         graph_test=graph_test,
#         y_train=y_train,
#         y_test=y_test,
#         batch_size=16,
#         lr=0.001,
#         epochs=5
#     )
#
#     model.train()
#     model.test()