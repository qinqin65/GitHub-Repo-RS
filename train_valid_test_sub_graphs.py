""" construct the train, validation and test sub knowledge graph """

import dgl
import pickle
import numpy as np
import os
import scipy.sparse as sp

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

    train_sub_graph = g0.clone()
    train_sub_graph.remove_edges(watch_eids[np.invert(watch_train_mask)], etype=('user', 'watch', 'repo'))
    train_sub_graph.remove_edges(watch_eids[np.invert(watch_train_mask)], etype=('repo', 'watched-by', 'user'))
    train_sub_graph.remove_edges(star_eids[np.invert(star_train_mask)], etype=('user', 'star', 'repo'))
    train_sub_graph.remove_edges(star_eids[np.invert(star_train_mask)], etype=('repo', 'starred-by', 'user'))
    train_sub_graph.remove_edges(fork_eids[np.invert(fork_train_mask)], etype=('user', 'fork', 'repo'))
    train_sub_graph.remove_edges(fork_eids[np.invert(fork_train_mask)], etype=('repo', 'forked-by', 'user'))
    train_sub_graph.remove_edges(own_eids[np.invert(own_train_mask)], etype=('user', 'own', 'repo'))
    train_sub_graph.remove_edges(own_eids[np.invert(own_train_mask)], etype=('repo', 'owned-by', 'user'))

    valid_sub_graph = g0.clone()
    valid_sub_graph.remove_edges(watch_eids[np.invert(watch_valid_mask)], etype=('user', 'watch', 'repo'))
    valid_sub_graph.remove_edges(watch_eids[np.invert(watch_valid_mask)], etype=('repo', 'watched-by', 'user'))
    valid_sub_graph.remove_edges(star_eids[np.invert(star_valid_mask)], etype=('user', 'star', 'repo'))
    valid_sub_graph.remove_edges(star_eids[np.invert(star_valid_mask)], etype=('repo', 'starred-by', 'user'))
    valid_sub_graph.remove_edges(fork_eids[np.invert(fork_valid_mask)], etype=('user', 'fork', 'repo'))
    valid_sub_graph.remove_edges(fork_eids[np.invert(fork_valid_mask)], etype=('repo', 'forked-by', 'user'))
    valid_sub_graph.remove_edges(own_eids[np.invert(own_valid_mask)], etype=('user', 'own', 'repo'))
    valid_sub_graph.remove_edges(own_eids[np.invert(own_valid_mask)], etype=('repo', 'owned-by', 'user'))

    test_sub_graph = g0.clone()
    test_sub_graph.remove_edges(watch_eids[np.invert(watch_test_mask)], etype=('user', 'watch', 'repo'))
    test_sub_graph.remove_edges(watch_eids[np.invert(watch_test_mask)], etype=('repo', 'watched-by', 'user'))
    test_sub_graph.remove_edges(star_eids[np.invert(star_test_mask)], etype=('user', 'star', 'repo'))
    test_sub_graph.remove_edges(star_eids[np.invert(star_test_mask)], etype=('repo', 'starred-by', 'user'))
    test_sub_graph.remove_edges(fork_eids[np.invert(fork_test_mask)], etype=('user', 'fork', 'repo'))
    test_sub_graph.remove_edges(fork_eids[np.invert(fork_test_mask)], etype=('repo', 'forked-by', 'user'))
    test_sub_graph.remove_edges(own_eids[np.invert(own_test_mask)], etype=('user', 'own', 'repo'))
    test_sub_graph.remove_edges(own_eids[np.invert(own_test_mask)], etype=('repo', 'owned-by', 'user'))

    # positive graphs and negative graphs
    # watch
    adj_watch = sp.coo_matrix((np.ones(len(watch_edges[0])), (watch_edges[0].numpy(), watch_edges[1].numpy())))
    adj_watch_neg = 1 - adj_watch.todense()
    neg_watch_u, neg_watch_v = np.where(adj_watch_neg != 0)

    neg_watch_eids = np.random.choice(len(neg_watch_u), g0.num_edges(('user', 'watch', 'repo')))
    chosen_neg_watch_u, chosen_neg_watch_v = neg_watch_u[neg_watch_eids], neg_watch_v[neg_watch_eids]

    # star
    adj_star = sp.coo_matrix((np.ones(len(star_edges[0])), (star_edges[0].numpy(), star_edges[1].numpy())))
    adj_star_neg = 1 - adj_star.todense()
    neg_star_u, neg_star_v = np.where(adj_star_neg != 0)

    neg_star_eids = np.random.choice(len(neg_star_u), g0.num_edges(('user', 'star', 'repo')))
    chosen_neg_star_u, chosen_neg_star_v = neg_star_u[neg_star_eids], neg_star_v[neg_star_eids]

    # fork
    adj_fork = sp.coo_matrix((np.ones(len(fork_edges[0])), (fork_edges[0].numpy(), fork_edges[1].numpy())))
    adj_fork_neg = 1 - adj_fork.todense()
    neg_fork_u, neg_fork_v = np.where(adj_fork_neg != 0)

    neg_fork_eids = np.random.choice(len(neg_fork_u), g0.num_edges(('user', 'fork', 'repo')))
    chosen_neg_fork_u, chosen_neg_fork_v = neg_fork_u[neg_fork_eids], neg_fork_v[neg_fork_eids]

    # own
    adj_own = sp.coo_matrix((np.ones(len(own_edges[0])), (own_edges[0].numpy(), own_edges[1].numpy())))
    adj_own_neg = 1 - adj_own.todense()
    neg_own_u, neg_own_v = np.where(adj_own_neg != 0)

    neg_own_eids = np.random.choice(len(neg_own_u), g0.num_edges(('user', 'own', 'repo')))
    chosen_neg_own_u, chosen_neg_own_v = neg_own_u[neg_own_eids], neg_own_v[neg_own_eids]
    
    num_nodes_dict = { 'user': g0.num_nodes('user'), 'repo': g0.num_nodes('repo') }

    # train
    train_sub_pos_g = dgl.heterograph({
        ('user', 'star', 'repo'): (star_edges[0][star_train_mask], star_edges[1][star_train_mask]),
        ('repo', 'starred-by', 'user'): (star_edges[1][star_train_mask], star_edges[0][star_train_mask]),
        ('user', 'watch', 'repo'): (watch_edges[0][watch_train_mask], watch_edges[1][watch_train_mask]),
        ('repo', 'watched-by', 'user'): (watch_edges[1][watch_train_mask], watch_edges[0][watch_train_mask]),
        ('user', 'fork', 'repo'): (fork_edges[0][fork_train_mask], fork_edges[1][fork_train_mask]),
        ('repo', 'forked-by', 'user'): (fork_edges[1][fork_train_mask], fork_edges[0][fork_train_mask]),
        ('user', 'own', 'repo'): (own_edges[0][own_train_mask], own_edges[1][own_train_mask]),
        ('repo', 'owned-by', 'user'): (own_edges[1][own_train_mask], own_edges[0][own_train_mask])
    }, num_nodes_dict=num_nodes_dict)

    train_sub_neg_g = dgl.heterograph({
        ('user', 'star', 'repo'): (chosen_neg_star_u[star_train_mask], chosen_neg_star_v[star_train_mask]),
        ('repo', 'starred-by', 'user'): (chosen_neg_star_v[star_train_mask], chosen_neg_star_u[star_train_mask]),
        ('user', 'watch', 'repo'): (chosen_neg_watch_u[watch_train_mask], chosen_neg_watch_v[watch_train_mask]),
        ('repo', 'watched-by', 'user'): (chosen_neg_watch_v[watch_train_mask], chosen_neg_watch_u[watch_train_mask]),
        ('user', 'fork', 'repo'): (chosen_neg_fork_u[fork_train_mask], chosen_neg_fork_v[fork_train_mask]),
        ('repo', 'forked-by', 'user'): (chosen_neg_fork_v[fork_train_mask], chosen_neg_fork_u[fork_train_mask]),
        ('user', 'own', 'repo'): (chosen_neg_own_u[own_train_mask], chosen_neg_own_v[own_train_mask]),
        ('repo', 'owned-by', 'user'): (chosen_neg_own_v[own_train_mask], chosen_neg_own_u[own_train_mask])
    }, num_nodes_dict=num_nodes_dict)

    # valid
    valid_sub_pos_g = dgl.heterograph({
        ('user', 'star', 'repo'): (star_edges[0][star_valid_mask], star_edges[1][star_valid_mask]),
        ('repo', 'starred-by', 'user'): (star_edges[1][star_valid_mask], star_edges[0][star_valid_mask]),
        ('user', 'watch', 'repo'): (watch_edges[0][watch_valid_mask], watch_edges[1][watch_valid_mask]),
        ('repo', 'watched-by', 'user'): (watch_edges[1][watch_valid_mask], watch_edges[0][watch_valid_mask]),
        ('user', 'fork', 'repo'): (fork_edges[0][fork_valid_mask], fork_edges[1][fork_valid_mask]),
        ('repo', 'forked-by', 'user'): (fork_edges[1][fork_valid_mask], fork_edges[0][fork_valid_mask]),
        ('user', 'own', 'repo'): (own_edges[0][own_valid_mask], own_edges[1][own_valid_mask]),
        ('repo', 'owned-by', 'user'): (own_edges[1][own_valid_mask], own_edges[0][own_valid_mask])
    }, num_nodes_dict=num_nodes_dict)

    valid_sub_neg_g = dgl.heterograph({
        ('user', 'star', 'repo'): (chosen_neg_star_u[star_valid_mask], chosen_neg_star_v[star_valid_mask]),
        ('repo', 'starred-by', 'user'): (chosen_neg_star_v[star_valid_mask], chosen_neg_star_u[star_valid_mask]),
        ('user', 'watch', 'repo'): (chosen_neg_watch_u[watch_valid_mask], chosen_neg_watch_v[watch_valid_mask]),
        ('repo', 'watched-by', 'user'): (chosen_neg_watch_v[watch_valid_mask], chosen_neg_watch_u[watch_valid_mask]),
        ('user', 'fork', 'repo'): (chosen_neg_fork_u[fork_valid_mask], chosen_neg_fork_v[fork_valid_mask]),
        ('repo', 'forked-by', 'user'): (chosen_neg_fork_v[fork_valid_mask], chosen_neg_fork_u[fork_valid_mask]),
        ('user', 'own', 'repo'): (chosen_neg_own_u[own_valid_mask], chosen_neg_own_v[own_valid_mask]),
        ('repo', 'owned-by', 'user'): (chosen_neg_own_v[own_valid_mask], chosen_neg_own_u[own_valid_mask])
    }, num_nodes_dict=num_nodes_dict)

    # test
    test_sub_pos_g = dgl.heterograph({
        ('user', 'star', 'repo'): (star_edges[0][star_test_mask], star_edges[1][star_test_mask]),
        ('repo', 'starred-by', 'user'): (star_edges[1][star_test_mask], star_edges[0][star_test_mask]),
        ('user', 'watch', 'repo'): (watch_edges[0][watch_test_mask], watch_edges[1][watch_test_mask]),
        ('repo', 'watched-by', 'user'): (watch_edges[1][watch_test_mask], watch_edges[0][watch_test_mask]),
        ('user', 'fork', 'repo'): (fork_edges[0][fork_test_mask], fork_edges[1][fork_test_mask]),
        ('repo', 'forked-by', 'user'): (fork_edges[1][fork_test_mask], fork_edges[0][fork_test_mask]),
        ('user', 'own', 'repo'): (own_edges[0][own_test_mask], own_edges[1][own_test_mask]),
        ('repo', 'owned-by', 'user'): (own_edges[1][own_test_mask], own_edges[0][own_test_mask])
    }, num_nodes_dict=num_nodes_dict)

    test_sub_neg_g = dgl.heterograph({
        ('user', 'star', 'repo'): (chosen_neg_star_u[star_test_mask], chosen_neg_star_v[star_test_mask]),
        ('repo', 'starred-by', 'user'): (chosen_neg_star_v[star_test_mask], chosen_neg_star_u[star_test_mask]),
        ('user', 'watch', 'repo'): (chosen_neg_watch_u[watch_test_mask], chosen_neg_watch_v[watch_test_mask]),
        ('repo', 'watched-by', 'user'): (chosen_neg_watch_v[watch_test_mask], chosen_neg_watch_u[watch_test_mask]),
        ('user', 'fork', 'repo'): (chosen_neg_fork_u[fork_test_mask], chosen_neg_fork_v[fork_test_mask]),
        ('repo', 'forked-by', 'user'): (chosen_neg_fork_v[fork_test_mask], chosen_neg_fork_u[fork_test_mask]),
        ('user', 'own', 'repo'): (chosen_neg_own_u[own_test_mask], chosen_neg_own_v[own_test_mask]),
        ('repo', 'owned-by', 'user'): (chosen_neg_own_v[own_test_mask], chosen_neg_own_u[own_test_mask])
    }, num_nodes_dict=num_nodes_dict)

    return (
        train_sub_graph, 
        valid_sub_graph, 
        test_sub_graph, 
        train_sub_pos_g, 
        train_sub_neg_g, 
        valid_sub_pos_g,
        valid_sub_neg_g,
        test_sub_pos_g,
        test_sub_neg_g
    )
        

if __name__ == "__main__":
    (
        train_sub_graph, 
        valid_sub_graph, 
        test_sub_graph,
        train_sub_pos_g, 
        train_sub_neg_g, 
        valid_sub_pos_g,
        valid_sub_neg_g,
        test_sub_pos_g,
        test_sub_neg_g
    ) = construct_sub_knowledge_graph()

    dgl.save_graphs('./data/sub_kowledge_graph.bin', [
        train_sub_graph, 
        valid_sub_graph, 
        test_sub_graph,
        train_sub_pos_g, 
        train_sub_neg_g, 
        valid_sub_pos_g,
        valid_sub_neg_g,
        test_sub_pos_g,
        test_sub_neg_g
    ])

