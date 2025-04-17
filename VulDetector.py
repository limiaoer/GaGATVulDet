import numpy as np
from parser import parameter_parser
from models.GeGAT import GeGATModel
from models.GCN import GNNModel
from preprocessing import get_graph_feature

args = parameter_parser()


def main():
    graph_train, graph_test, graph_experts_train, graph_experts_test = get_graph_feature()

    graph_train = np.array(graph_train)
    graph_test = np.array(graph_test)

    y_train = []
    for i in range(len(graph_experts_train)):
        y_train.append(int(graph_experts_train[i]))
    y_train = np.array(y_train)

    y_test = []
    for i in range(len(graph_experts_test)):
        y_test.append(int(graph_experts_test[i]))
    y_test = np.array(y_test)

    if args.model == 'GeGAT':
        model = GeGATModel(graph_train, graph_test,  y_train, y_test)
    elif args.model == 'GCN':
        model = GNNModel(graph_train, graph_test,  y_train, y_test)

    model.train()
    model.test()


if __name__ == "__main__":
    main()
