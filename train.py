import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from KGCN import Model, loss_fn

def process_graph(graph: dgl.heterograph):
    number_of_users = graph.num_nodes('user')
    number_of_repos = graph.num_nodes('repo')
    ground_truth = torch.zeros((number_of_users, number_of_repos), dtype=torch.int8)
    for etype in graph.canonical_etypes:
        # ignore the reverse relation
        if etype[0] != 'user' and etype[1] != 'repo':
            continue
        # sample 40% for testing
        number_of_edges = graph.number_of_edges(etype=etype)
        train_number = round(number_of_edges * 0.6)
        test_number = round(number_of_edges * 0.4)
        sample_indexes = np.array([0] * train_number + [1] * test_number)
        np.random.shuffle(sample_indexes)

        train_mask = sample_indexes==0
        test_mask = sample_indexes==1

        edges = graph.edges(etype=etype)
        edge_ids = graph.edge_ids(edges[0], edges[1], etype=etype)

        graph.remove_edges(edge_ids[test_mask], etype=etype)

        ground_truth[edges[0][test_mask], edges[1][test_mask]] = 1
    
    return graph, ground_truth

def train():
    EPOCH = 100
    TOP_K = 10
    neg_sample_size = 5

    g, l = dgl.load_graphs('./data/sub_kowledge_graph.bin')
    train_graph = g[0]
    valid_graph = g[1]
    test_graph = g[2]
    
    model = Model(train_graph, 150, 261, 50)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    edges_fork = train_graph.edges(etype=('user', 'fork', 'repo'))
    edges_own = train_graph.edges(etype=('user', 'own', 'repo'))
    edges_star = train_graph.edges(etype=('user', 'star', 'repo'))
    edges_watch = train_graph.edges(etype=('user', 'watch', 'repo'))
    train_eid = {('user', 'fork', 'repo'): train_graph.edge_ids(edges_fork[0], edges_fork[1], etype=('user', 'fork', 'repo')),
        ('user', 'own', 'repo'): train_graph.edge_ids(edges_own[0], edges_own[1], etype=('user', 'own', 'repo')),
        ('user', 'star', 'repo'): train_graph.edge_ids(edges_star[0], edges_star[1], etype=('user', 'star', 'repo')),
        ('user', 'watch', 'repo'): train_graph.edge_ids(edges_watch[0], edges_watch[1], etype=('user', 'watch', 'repo'))}
    dataloader = dgl.dataloading.EdgeDataLoader(
    train_graph, train_eid, sampler,
    exclude='reverse_types',
    reverse_etypes={'watch': 'watched-by', 'watched-by': 'watch',
                    'star': 'starred-by', 'starred-by': 'star',
                    'fork': 'forked-by', 'forked-by': 'fork',
                    'own': 'owned-by', 'owned-by': 'own'},
    negative_sampler=dgl.dataloading.negative_sampler.Uniform(neg_sample_size),
    batch_size=1024, shuffle=True, drop_last=False)

    processed_valid_graph, ground_truth_valid_data = process_graph(valid_graph)
    processed_test_graph, ground_truth_test_data = process_graph(test_graph)

    for epoch in range(EPOCH):
        training_loops = 0
        total_loss = 0
        for input_nodes, pos_g, neg_g, blocks in dataloader:
            user_feat = blocks[0].ndata['graph_data']['user']
            repo_feat = blocks[0].ndata['graph_data']['repo']
            model.train()
            pos_score, neg_score = model(blocks, pos_g, neg_g, user_feat, repo_feat)
            loss = loss_fn(pos_score, neg_score, neg_sample_size)

            total_loss += loss.item()
            training_loops += 1

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_avg_loss = total_loss / training_loops

        if epoch % 5 == 0:
            # valid top k recommendation
            valid_mean_hit_rate = 0
            model.eval()
            with torch.no_grad():
                hit_rates = np.zeros(valid_graph.num_nodes('user'))
         
                h_user = model.user_embedding(processed_valid_graph.ndata['graph_data']['user'])
                h_repo = model.repo_embedding(processed_valid_graph.ndata['graph_data']['repo'])

                h_dict = {
                    'user': h_user,
                    'repo': h_repo
                }

                h = model.hidden(processed_valid_graph, h_dict)
                out = model.out(processed_valid_graph, h)

                user_embedding = out['user']
                repo_embedding = out['repo']

                user_emb_normalized = F.normalize(user_embedding)
                repo_emb_normalized = F.normalize(repo_embedding)

                user_repo_rating = user_emb_normalized @ repo_emb_normalized.T
                for i, rating in enumerate(user_repo_rating):
                    recommendation = rating.detach().numpy()
                    recommendation = recommendation.argsort()[-TOP_K:]
                    ground_truth = np.where(ground_truth_valid_data[i]>0)[0]

                    recommendation_set = set(recommendation)
                    ground_truth_set = set(ground_truth)

                    intersections = recommendation_set.intersection(ground_truth_set)
                    hit_rate = 0 if len(ground_truth_set) == 0 else len(intersections) / min(len(ground_truth_set), TOP_K)
                    hit_rates[i] = min(hit_rate, 1)
                valid_mean_hit_rate = np.mean(hit_rates)
            
            # test top k recommendation
            test_mean_hit_rate = 0
            model.eval()
            with torch.no_grad():
                hit_rates = np.zeros(test_graph.num_nodes('user'))
         
                h_user = model.user_embedding(processed_test_graph.ndata['graph_data']['user'])
                h_repo = model.repo_embedding(processed_test_graph.ndata['graph_data']['repo'])

                h_dict = {
                    'user': h_user,
                    'repo': h_repo
                }

                h = model.hidden(processed_test_graph, h_dict)
                out = model.out(processed_test_graph, h)

                user_embedding = out['user']
                repo_embedding = out['repo']

                user_emb_normalized = F.normalize(user_embedding)
                repo_emb_normalized = F.normalize(repo_embedding)

                user_repo_rating = user_emb_normalized @ repo_emb_normalized.T
                for i, rating in enumerate(user_repo_rating):
                    recommendation = rating.detach().numpy()
                    recommendation = recommendation.argsort()[-TOP_K:]
                    ground_truth = np.where(ground_truth_test_data[i]>0)[0]

                    recommendation_set = set(recommendation)
                    ground_truth_set = set(ground_truth)

                    intersections = recommendation_set.intersection(ground_truth_set)
                    hit_rate = 0 if len(ground_truth_set) == 0 else len(intersections) / min(len(ground_truth_set), TOP_K)
                    hit_rates[i] = min(hit_rate, 1)
                test_mean_hit_rate = np.mean(hit_rates)

            print('In epoch {}, loss: {:.3f}, valid hit rate: {:.3f}, test hit rate: {:.3f}'.format(
                epoch, train_avg_loss, valid_mean_hit_rate, test_mean_hit_rate))

if __name__ == '__main__':
    train()