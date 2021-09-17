""" construct the train, validation and test sub knowledge graph """

import dgl
import pickle
import numpy as np
import os

def construct_sub_knowledge_graph():
    g, l = dgl.load_graphs('./data/saved_kowledge_graph.bin')
    g0 = g[0]

    # generate the train, validation and test mask
    # watch
    watch_number_of_edges = g0.number_of_edges(etype=('user', 'watch', 'repo'))
    watch_train_number = round(watch_number_of_edges * 0.6)
    watch_valid_number = round(watch_number_of_edges * 0.2)
    watch_test_number = watch_number_of_edges - watch_train_number - watch_valid_number
    watch_sample_indexes = np.array([0] * watch_train_number + [1] * watch_valid_number + [2] * watch_test_number)
    np.random.shuffle(watch_sample_indexes)

    watch_train_mask = watch_sample_indexes==0
    watch_valid_mask = watch_sample_indexes==1
    watch_test_mask = watch_sample_indexes==2

    # star
    star_number_of_edges = g0.number_of_edges(etype=('user', 'star', 'repo'))
    star_train_number = round(star_number_of_edges * 0.6)
    star_valid_number = round(star_number_of_edges * 0.2)
    star_test_number = star_number_of_edges - star_train_number - star_valid_number
    star_sample_indexes = np.array([0] * star_train_number + [1] * star_valid_number + [2] * star_test_number)
    np.random.shuffle(star_sample_indexes)

    star_train_mask = star_sample_indexes==0
    star_valid_mask = star_sample_indexes==1
    star_test_mask = star_sample_indexes==2

    # fork
    fork_number_of_edges = g0.number_of_edges(etype=('user', 'fork', 'repo'))
    fork_train_number = round(fork_number_of_edges * 0.6)
    fork_valid_number = round(fork_number_of_edges * 0.2)
    fork_test_number = fork_number_of_edges - fork_train_number - fork_valid_number
    fork_sample_indexes = np.array([0] * fork_train_number + [1] * fork_valid_number + [2] * fork_test_number)
    np.random.shuffle(fork_sample_indexes)

    fork_train_mask = fork_sample_indexes==0
    fork_valid_mask = fork_sample_indexes==1
    fork_test_mask = fork_sample_indexes==2

    # own
    own_number_of_edges = g0.number_of_edges(etype=('user', 'own', 'repo'))
    own_train_number = round(own_number_of_edges * 0.6)
    own_valid_number = round(own_number_of_edges * 0.2)
    own_test_number = own_number_of_edges - own_train_number - own_valid_number
    own_sample_indexes = np.array([0] * own_train_number + [1] * own_valid_number + [2] * own_test_number)
    np.random.shuffle(own_sample_indexes)

    own_train_mask = own_sample_indexes==0
    own_valid_mask = own_sample_indexes==1
    own_test_mask = own_sample_indexes==2

    # edge ids
    watch_edges = g0.edges(etype=('user', 'watch', 'repo'))
    watch_eids = g0.edge_ids(watch_edges[0], watch_edges[1], etype=('user', 'watch', 'repo'))

    star_edges = g0.edges(etype=('user', 'star', 'repo'))
    star_eids = g0.edge_ids(star_edges[0], star_edges[1], etype=('user', 'star', 'repo'))

    fork_edges = g0.edges(etype=('user', 'fork', 'repo'))
    fork_eids = g0.edge_ids(fork_edges[0], fork_edges[1], etype=('user', 'fork', 'repo'))

    own_edges = g0.edges(etype=('user', 'own', 'repo'))
    own_eids = g0.edge_ids(own_edges[0], own_edges[1], etype=('user', 'own', 'repo'))

    train_sub_graph = dgl.edge_subgraph(g0, {
        ('user', 'watch', 'repo'): watch_eids[watch_train_mask],
        ('repo', 'watched-by', 'user'): watch_eids[watch_train_mask], 
        ('user', 'star', 'repo'): star_eids[star_train_mask],
        ('repo', 'starred-by', 'user'): star_eids[star_train_mask],
        ('user', 'fork', 'repo'): fork_eids[fork_train_mask],
        ('repo', 'forked-by', 'user'): fork_eids[fork_train_mask],
        ('user', 'own', 'repo'): own_eids[own_train_mask],
        ('repo', 'owned-by', 'user'): own_eids[own_train_mask]
    })
    valid_sub_graph = dgl.edge_subgraph(g0, {
        ('user', 'watch', 'repo'): watch_eids[watch_valid_mask],
        ('repo', 'watched-by', 'user'): watch_eids[watch_valid_mask], 
        ('user', 'star', 'repo'): star_eids[star_valid_mask],
        ('repo', 'starred-by', 'user'): star_eids[star_valid_mask],
        ('user', 'fork', 'repo'): fork_eids[fork_valid_mask],
        ('repo', 'forked-by', 'user'): fork_eids[fork_valid_mask],
        ('user', 'own', 'repo'): own_eids[own_valid_mask],
        ('repo', 'owned-by', 'user'): own_eids[own_valid_mask]
    })
    test_sub_graph = dgl.edge_subgraph(g0, {
        ('user', 'watch', 'repo'): watch_eids[watch_test_mask], 
        ('repo', 'watched-by', 'user'): watch_eids[watch_test_mask],
        ('user', 'star', 'repo'): star_eids[star_test_mask],
        ('repo', 'starred-by', 'user'): star_eids[star_test_mask],
        ('user', 'fork', 'repo'): fork_eids[fork_test_mask],
        ('repo', 'forked-by', 'user'): fork_eids[fork_test_mask],
        ('user', 'own', 'repo'): own_eids[own_test_mask],
        ('repo', 'owned-by', 'user'): own_eids[own_test_mask]
    })

    return train_sub_graph, valid_sub_graph, test_sub_graph

if __name__ == "__main__":
    train_sub_graph, valid_sub_graph, test_sub_graph = construct_sub_knowledge_graph()
    dgl.save_graphs('./data/sub_kowledge_graph.bin', [train_sub_graph, valid_sub_graph, test_sub_graph])

