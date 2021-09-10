import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from KGCN import Model, loss_fn

def train():
    EPOCH = 100
    TOP_K = 10
    neg_sample_size = 5

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
    
    labels = torch.zeros((number_of_users, number_of_repos))
    tests = torch.zeros((number_of_users, number_of_repos))

    for interaction, index in interactions_map.items():
        edge = g0.edges(etype=interaction)
        labels[edge[0], edge[1]] = 1
        tests[edge[0], edge[1]] = interactions_map[interaction] + 1

    labels = torch.flatten(labels)
    
    model = Model(g0, 150, 261, 50)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    edges_fork = g0.edges(etype=('user', 'fork', 'repo'))
    edges_own = g0.edges(etype=('user', 'own', 'repo'))
    edges_star = g0.edges(etype=('user', 'star', 'repo'))
    edges_watch = g0.edges(etype=('user', 'watch', 'repo'))
    train_eid = {('user', 'fork', 'repo'): g0.edge_ids(edges_fork[0], edges_fork[1], etype=('user', 'fork', 'repo')),
        ('user', 'own', 'repo'): g0.edge_ids(edges_own[0], edges_own[1], etype=('user', 'own', 'repo')),
        ('user', 'star', 'repo'): g0.edge_ids(edges_star[0], edges_star[1], etype=('user', 'star', 'repo')),
        ('user', 'watch', 'repo'): g0.edge_ids(edges_watch[0], edges_watch[1], etype=('user', 'watch', 'repo'))}
    dataloader = dgl.dataloading.EdgeDataLoader(
    g0, train_eid, sampler,
    negative_sampler=dgl.dataloading.negative_sampler.Uniform(neg_sample_size),
    batch_size=1024, shuffle=True, drop_last=False)

    best_val_acc = 0
    best_test_acc = 0

    for epoch in range(EPOCH):
        i = 0
        total_loss = 0
        for input_nodes, pos_g, neg_g, blocks in dataloader:
            user_feat = blocks[0].ndata['graph_data']['user']
            repo_feat = blocks[0].ndata['graph_data']['repo']
            pos_score, neg_score = model(blocks, pos_g, neg_g, user_feat, repo_feat)
            loss = loss_fn(pos_score, neg_score, neg_sample_size)

            total_loss += loss.item()

            # top k recommendation
            # hit_rates = np.zeros(number_of_users)
            # user_repo_rating = logits.reshape(number_of_users, number_of_repos)
            # for i, rating in enumerate(user_repo_rating):
            #     recommendation = rating.detach().numpy()
            #     recommendation = recommendation.argsort()[-TOP_K:]
            #     ground_truth = np.where(tests[i]>0)[0]

            #     recommendation_set = set(recommendation)
            #     ground_truth_set = set(ground_truth)

            #     intersections = recommendation_set.intersection(ground_truth_set)
            #     hit_rate = 0 if len(ground_truth_set) == 0 else len(intersections) / max(len(ground_truth_set), TOP_K)
            #     hit_rates[i] = min(hit_rate, 1)
            # mean_hit_rate = np.mean(hit_rates)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_avg_loss = total_loss / i

        if epoch % 5 == 0:
            print('In epoch {}, loss: {:.3f}, hit rate: {:.3f}'.format(
                epoch, train_avg_loss, 0))

if __name__ == '__main__':
    train()