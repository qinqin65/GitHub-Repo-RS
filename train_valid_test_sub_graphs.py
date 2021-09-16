""" construct the train, validation and test sub knowledge graph """

import dgl
import pickle
import numpy as np
import os

def construct_sub_knowledge_graph():
    users_id_map = pickle.load(open('./data/users_id_map.p', 'rb'))
    repos_id_map = pickle.load(open('./data/repos_id_map.p', 'rb'))
    interactions_map = pickle.load(open('./data/interactions_map.p', 'rb'))
    g, l = dgl.load_graphs('./data/saved_kowledge_graph.bin')
    g0 = g[0]

    # generate the train, validation and test mask
    users_count = len(users_id_map)
    train_number = round(users_count * 0.6)
    valid_number = round(users_count * 0.2)
    test_number = users_count - train_number - valid_number
    sample_indexes = np.array([0] * train_number + [1] * valid_number + [2] * test_number)
    np.random.shuffle(sample_indexes)

    train_mask = sample_indexes==0
    valid_mask = sample_indexes==1
    test_mask = sample_indexes==2

    user_ids = np.fromiter(users_id_map.values(), dtype=np.int32)
    repo_ids = np.fromiter(repos_id_map.values(), dtype=np.int32)

    train_sub_graph = dgl.node_subgraph(g0, {'user': user_ids[train_mask], 'repo': repo_ids})
    valid_sub_graph = dgl.node_subgraph(g0, {'user': user_ids[valid_mask], 'repo': repo_ids})
    test_sub_graph = dgl.node_subgraph(g0, {'user': user_ids[test_mask], 'repo': repo_ids})

    return train_sub_graph, valid_sub_graph, test_sub_graph

if __name__ == "__main__":
    train_sub_graph, valid_sub_graph, test_sub_graph = construct_sub_knowledge_graph()
    pickle.dump(train_sub_graph, open('./data/train_sub_graph.p', 'wb'))
    pickle.dump(valid_sub_graph, open('./data/valid_sub_graph.p', 'wb'))
    pickle.dump(test_sub_graph, open('./data/test_sub_graph.p', 'wb'))

