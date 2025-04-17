import numpy as np


def get_graph_feature():
    graph_feature_train_data_path = "./graph_feature/timestamp/timestamp_train_feature.txt"
    graph_feature_train_label_path = "./graph_feature/timestamp/label_by_experts_train.txt"

    graph_feature_test_data_path = "./graph_feature/timestamp/timestamp_valid_feature.txt"
    graph_feature_test_label_path = "./graph_feature/timestamp/label_by_experts_valid.txt"

    label_by_experts_train = []
    f_train_label_expert = open(graph_feature_train_label_path, 'r')
    labels = f_train_label_expert.readlines()
    for label in labels:
        label_by_experts_train.append(label.strip('\n'))

    label_by_experts_valid = []
    f_test_label_expert = open(graph_feature_test_label_path, 'r')
    labels = f_test_label_expert.readlines()
    for label in labels:
        label_by_experts_valid.append(label.strip('\n'))

    graph_feature_train = np.loadtxt(graph_feature_train_data_path).tolist()  # graph feature train
    graph_feature_test = np.loadtxt(graph_feature_test_data_path, delimiter=", ").tolist()  # graph feature test

    for i in range(len(graph_feature_train)):
        graph_feature_train[i] = [graph_feature_train[i]]

    for i in range(len(graph_feature_test)):
        graph_feature_test[i] = [graph_feature_test[i]]

    return graph_feature_train, graph_feature_test, label_by_experts_train, label_by_experts_valid


if __name__ == "__main__":
    graph_train, graph_test, graph_experts_train, graph_experts_test = get_graph_feature()
    print()

