import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from KGCN import Model

from numpy.core.defchararray import index

def train():
    EPOCH = 1000
    EMBEDDING_LENGTH = 400

    users_id_map = pickle.load(open('./data/users_id_map.p', 'rb'))
    repos_id_map = pickle.load(open('./data/repos_id_map.p', 'rb'))
    interactions_map = pickle.load(open('./data/interactions_map.p', 'rb'))
    entity_embedding = np.load('./data/results/TransR_GHRS_0/GHRS_TransR_entity.npy')
    relation_embedding = np.load('./data/results/TransR_GHRS_0/GHRS_TransR_relation.npy')
    projection = np.load('./data/results/TransR_GHRS_0/GHRS_TransRprojection.npy')
    g, l = dgl.load_graphs('./data/saved_kowledge_graph.bin')
    g0 = g[0]

    nodes_repo = g0.nodes('repo')
    # there are useless repository nodes which are disabled or archieved. So it needs to be removed.
    g0.remove_nodes(nodes_repo[7054:], ntype='repo')

    number_of_users = len(users_id_map)
    number_of_repos = len(repos_id_map)
    # get the number of edges
    number_of_edges = number_of_users * number_of_repos

    # sample subgraphs for train, validation and test with the ratio of 60:20:20
    train_number = round(number_of_edges * 0.6)
    valid_number = round(number_of_edges * 0.2)
    test_number = number_of_edges- train_number - valid_number
    sample_indexes = np.array([0] * train_number + [1] * valid_number + [2] * test_number)
    np.random.shuffle(sample_indexes)

    # generate the train, validation and test mask
    train_mask = sample_indexes==0
    valid_mask = sample_indexes==1
    test_mask = sample_indexes==2

    repo_id_shift = len(users_id_map)

    # fill the trained embedding data to the graph
    g0.nodes['user'].data['h'] = torch.from_numpy(entity_embedding[:repo_id_shift])
    g0.nodes['repo'].data['h'] = torch.from_numpy(entity_embedding[repo_id_shift:])
    
    # construct the model for each interaction
    for interaction, index in interactions_map.items():
        # g0.edges[interaction].data['e'] = torch.from_numpy(np.tile(relation_embedding[index], (g0.num_edges(etype=interaction), 1)))
        p = projection[index].reshape(EMBEDDING_LENGTH, EMBEDDING_LENGTH)
        
        g0.nodes['user'].data['h_%s' % interaction] = g0.nodes['user'].data['h'] @ p
        g0.nodes['repo'].data['h_%s' % interaction] = g0.nodes['repo'].data['h'] @ p
        
        model = Model(EMBEDDING_LENGTH, EMBEDDING_LENGTH, interaction)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        edge = g0.edges(etype=interaction)

        labels = torch.zeros((number_of_users, number_of_repos))
        labels[edge[0], edge[1]] = 1
        labels = torch.flatten(labels)

        best_val_acc = 0
        best_test_acc = 0

        print('Interaction: %s' % interaction)

        for epoch in range(EPOCH):
            logits = model(g0)
            loss = F.binary_cross_entropy_with_logits(logits[train_mask], labels[train_mask])

            logits[logits>0.5] = 1
            # Compute accuracy on training/validation/test
            train_acc = (logits[train_mask] == labels[train_mask]).float().mean()
            val_acc = (logits[valid_mask] == labels[valid_mask]).float().mean()
            test_acc = (logits[test_mask] == labels[test_mask]).float().mean()

            # Save the best validation accuracy and the corresponding test accuracy.
            if best_val_acc < val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 5 == 0:
                print('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(
                    epoch, loss, val_acc, best_val_acc, test_acc, best_test_acc))

if __name__ == '__main__':
    train()