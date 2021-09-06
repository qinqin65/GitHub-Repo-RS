import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from KGCN import Model

def train():
    EPOCH = 100
    TOP_K = 10

    users_id_map = pickle.load(open('./data/users_id_map.p', 'rb'))
    repos_id_map = pickle.load(open('./data/repos_id_map.p', 'rb'))
    interactions_map = pickle.load(open('./data/interactions_map.p', 'rb'))
    g, l = dgl.load_graphs('./data/saved_kowledge_graph.bin')
    g0 = g[0]

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

    user_feat = g0.ndata['graph_data']['user']
    repo_feat = g0.ndata['graph_data']['repo']
    
    labels = torch.zeros((number_of_users, number_of_repos))
    tests = torch.zeros((number_of_users, number_of_repos))

    for interaction, index in interactions_map.items():
        edge = g0.edges(etype=interaction)
        labels[edge[0], edge[1]] = 1
        tests[edge[0], edge[1]] = interactions_map[index] + 1

    labels = torch.flatten(labels)
    
    model = Model(g0, 150, 261, 50)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    best_val_acc = 0
    best_test_acc = 0

    for epoch in range(EPOCH):
        logits = model(g0, user_feat, repo_feat)
        loss = F.binary_cross_entropy_with_logits(logits[train_mask], labels[train_mask])

        logits[logits>0.5] = 1
        # Compute accuracy on training/validation/test
        train_acc = (logits[train_mask] == labels[train_mask]).float().mean()
        val_acc = (logits[valid_mask] == labels[valid_mask]).float().mean()
        test_acc = (logits[test_mask] == labels[test_mask]).float().mean()

        # top k recommendation
        hit_rates = np.zeros(number_of_users)
        user_repo_rating = logits.reshape(number_of_users, number_of_repos)
        for i, rating in enumerate(user_repo_rating):
            recommendation = rating.argsort()[-TOP_K:]
            ground_truth = tests.argsort()[-TOP_K:]

            recommendation_set = set(recommendation)
            ground_truth_set = set(ground_truth)

            intersections = recommendation_set.intersection(ground_truth)
            hit_rate = len(intersections) / len(ground_truth_set)
            hit_rates[i] = hit_rate
        mean_hit_rate = np.mean(hit_rates)

        # Save the best validation accuracy and the corresponding test accuracy.
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            print('In epoch {}, loss: {:.3f}, hit rate: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(
                epoch, loss, mean_hit_rate, val_acc, best_val_acc, test_acc, best_test_acc))

if __name__ == '__main__':
    train()