""" construct the knowledge graph in (head, relation, tail) format """

import dgl
import pickle
import numpy as np
import os

def construct_knowledge_graph():
    users_id_map = pickle.load(open('./data/users_id_map.p', 'rb'))
    repos_id_map = pickle.load(open('./data/repos_id_map.p', 'rb'))
    interactions_map = pickle.load(open('./data/interactions_map.p', 'rb'))
    g, l = dgl.load_graphs('./data/saved_kowledge_graph.bin')
    g0 = g[0]

    relation_id_map = [] # construct relation ID map
    graph_hrt = []
    repo_id_shift = len(users_id_map) # to avoid id collision for user and repo

    for etype, eid in interactions_map.items():
        relation_id_map.append('%s,%s' % (eid, etype))
        edge = g0.edges(etype=etype)

        # generate the graph in the form of (head, relation, tail)
        for user, repo in zip(edge[0], edge[1]):
            graph_hrt.append('%s,%s,%s' % (user.item(), eid, repo.item() + repo_id_shift))
    
    # get the number of edges
    number_of_edges = g0.number_of_edges()
    assert number_of_edges == len(graph_hrt)
    
    # sample subgraphs for train, validation and test with the ratio of 60:20:20
    train_number = round(number_of_edges * 0.6)
    valid_number = round(number_of_edges * 0.2)
    test_number = number_of_edges- train_number - valid_number
    sample_indexes = np.array([0] * train_number + [1] * valid_number + [2] * test_number)
    np.random.shuffle(sample_indexes)

    # construct train, valid and test data set
    graph_hrt = np.array(graph_hrt)
    graph_hrt_train = graph_hrt[sample_indexes==0]
    graph_hrt_valid = graph_hrt[sample_indexes==1]
    graph_hrt_test = graph_hrt[sample_indexes==2]
    np.random.shuffle(graph_hrt_train)
    np.random.shuffle(graph_hrt_valid)
    np.random.shuffle(graph_hrt_test)

    # construct entity ID map
    # id_users_map = {v: k for k, v in users_id_map.items()}
    # id_repos_map = {v: k for k, v in repos_id_map.items()}
    id_users_map = ['%s,%s' % (v, k) for k, v in users_id_map.items()]
    id_repos_map = ['%s,%s' % (v + repo_id_shift, k) for k, v in repos_id_map.items()]
    entity_id_map = id_users_map + id_repos_map

    return graph_hrt_train, graph_hrt_valid, graph_hrt_test, entity_id_map, relation_id_map

if __name__ == "__main__":
    graph_hrt_train, graph_hrt_valid, graph_hrt_test, entity_id_map, relation_id_map = construct_knowledge_graph()
    
    f = open('./data/graph_hrt_train.txt', 'w')
    f.write(os.linesep.join(graph_hrt_train))
    f.close()

    f = open('./data/graph_hrt_valid.txt', 'w')
    f.write(os.linesep.join(graph_hrt_valid))
    f.close()

    f = open('./data/graph_hrt_test.txt', 'w')
    f.write(os.linesep.join(graph_hrt_test))
    f.close()

    f = open('./data/entities.dict', 'w')
    f.write(os.linesep.join(entity_id_map))
    f.close()

    f = open('./data/relations.dict', 'w')
    f.write(os.linesep.join(relation_id_map))
    f.close()