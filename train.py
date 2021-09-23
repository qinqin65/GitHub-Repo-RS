import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from KGCN import Model, loss_fn
import pickle

def process_graph(graph: dgl.heterograph):
    number_of_users = graph.num_nodes('user')
    number_of_repos = graph.num_nodes('repo')
    ground_truth = torch.zeros((number_of_users, number_of_repos), dtype=torch.int8)
    trained_truth = torch.zeros((number_of_users, number_of_repos), dtype=torch.int8)
    user_repo = torch.zeros((number_of_users, number_of_repos), dtype=torch.int8)
    graph_clone = graph.clone()
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

        graph_clone.remove_edges(edge_ids[test_mask], etype=etype)

        trained_truth[edges[0][train_mask], edges[1][train_mask]] = 1
        ground_truth[edges[0][test_mask], edges[1][test_mask]] = 1
        user_repo[edges[0], edges[1]] = 1
    
    repos_per_user = torch.sum(user_repo, axis=1)

    return graph_clone, ground_truth, trained_truth, repos_per_user.numpy()

def train():
    EPOCH = 100
    TOP_K = 10
    NEG_SAMPLE_SIZE = 5
    USER_INPUT_SIZE = 150
    REPO_INPUT_SIZE = 361
    USER_REPO_OUTPUT_SIZE = 125
    HIDDEN_OUTPUT_SIZE = 96
    OUT_SIZE = 50

    g, l = dgl.load_graphs('./data/sub_kowledge_graph.bin')
    train_graph = g[0]
    valid_graph = g[1]
    test_graph = g[2]
    interaction_matrix = pickle.load(open('data/interaction_matrix.p', 'rb'))
    
    model = Model(train_graph, USER_INPUT_SIZE, REPO_INPUT_SIZE, USER_REPO_OUTPUT_SIZE, HIDDEN_OUTPUT_SIZE, OUT_SIZE)
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
    negative_sampler=dgl.dataloading.negative_sampler.Uniform(NEG_SAMPLE_SIZE),
    batch_size=1024, shuffle=True, drop_last=False)

    processed_train_graph, ground_truth_train_data, trained_truth_train_data, repos_per_user_train = process_graph(train_graph)
    processed_valid_graph, ground_truth_valid_data, trained_truth_valid_data, repos_per_user_valid = process_graph(valid_graph)
    processed_test_graph, ground_truth_test_data, trained_truth_test_data, repos_per_user_test = process_graph(test_graph)

    for epoch in range(EPOCH):
        training_loops = 0
        total_loss = 0
        for input_nodes, pos_g, neg_g, blocks in dataloader:
            user_feat = blocks[0].ndata['graph_data']['user']
            repo_feat = blocks[0].ndata['graph_data']['repo']
            model.train()
            pos_score, neg_score = model(blocks, pos_g, neg_g, user_feat, repo_feat)
            loss = loss_fn(pos_score, neg_score, NEG_SAMPLE_SIZE)

            total_loss += loss.item()
            training_loops += 1

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_avg_loss = total_loss / training_loops

        if epoch % 5 == 0:
            # train top k recommendation
            model.eval()
            with torch.no_grad():
                h_user = model.user_embedding(processed_train_graph.ndata['graph_data']['user'])
                h_repo = model.repo_embedding(processed_train_graph.ndata['graph_data']['repo'])

                h_dict = {
                    'user': h_user,
                    'repo': h_repo
                }

                h = model.hidden(train_graph, h_dict)
                out = model.out(train_graph, h)

                user_embedding = out['user']
                repo_embedding = out['repo']

                user_emb_normalized = F.normalize(user_embedding)
                repo_emb_normalized = F.normalize(repo_embedding)

                user_repo_rating = user_emb_normalized @ repo_emb_normalized.T
                hit_rates = np.zeros(train_graph.num_nodes('user'))
                for i, rating in enumerate(user_repo_rating):
                    recommendation = rating[trained_truth_train_data[i]<1].detach().numpy()
                    recommendation = recommendation.argsort()[-TOP_K:]
                    ground_truth = np.where(ground_truth_train_data[i]>0)[0]
                    if len(ground_truth) < TOP_K:
                        hit_rates[i] = -1
                        continue

                    recommendation_set = set(recommendation)
                    ground_truth_set = set(ground_truth)

                    intersections = recommendation_set.intersection(ground_truth_set)
                    number_of_intersections = len(intersections)
                    number_of_ground_truth = len(ground_truth_set)
                    hit_rate = -1 if number_of_ground_truth == 0 else number_of_intersections / min(number_of_ground_truth, TOP_K)
                    hit_rates[i] = min(hit_rate, 1)
                mean_hit_rate = np.mean(hit_rates[hit_rates>-1])

            # # valid top k recommendation
            valid_mean_hit_rate = 0
            valid_group_0_5_hit_rate = 0
            valid_group_5_10_hit_rate = 0
            valid_group_10_15_hit_rate = 0
            valid_group_15_over_hit_rate = 0
            model.eval()
            with torch.no_grad():
                hit_rates = np.zeros(valid_graph.num_nodes('user'))
                group_0_5 = []
                group_5_10 = []
                group_10_15 = []
                group_15_over = []
         
                h_user = model.user_embedding(processed_valid_graph.ndata['graph_data']['user'])
                h_repo = model.repo_embedding(processed_valid_graph.ndata['graph_data']['repo'])

                h_dict = {
                    'user': h_user,
                    'repo': h_repo
                }

                h = model.hidden(valid_graph, h_dict)
                out = model.out(valid_graph, h)

                user_embedding = out['user']
                repo_embedding = out['repo']

                user_emb_normalized = F.normalize(user_embedding)
                repo_emb_normalized = F.normalize(repo_embedding)

                user_repo_rating = user_emb_normalized @ repo_emb_normalized.T
                for i, rating in enumerate(user_repo_rating):
                    recommendation = rating[trained_truth_valid_data[i]<1].detach().numpy()
                    recommendation = recommendation.argsort()[-TOP_K:]
                    ground_truth = np.where(ground_truth_valid_data[i]>0)[0]
                    if len(ground_truth) < TOP_K:
                        hit_rates[i] = -1
                        continue

                    recommendation_set = set(recommendation)
                    ground_truth_set = set(ground_truth)

                    intersections = recommendation_set.intersection(ground_truth_set)
                    number_of_intersections = len(intersections)
                    number_of_ground_truth = len(ground_truth_set)
                    hit_rate = -1 if number_of_ground_truth == 0 else number_of_intersections / min(number_of_ground_truth, TOP_K)
                    hit_rates[i] = min(hit_rate, 1)

                    # grouping
                    if repos_per_user_valid[i] < 5:
                        group_0_5.append(i)
                    elif repos_per_user_valid[i] < 10:
                        group_5_10.append(i)
                    elif repos_per_user_valid[i] < 15:
                        group_10_15.append(i)
                    else:
                        group_15_over.append(i)

                valid_mean_hit_rate = np.mean(hit_rates[hit_rates>-1])
                valid_group_0_5_hit_rate = np.mean(hit_rates[group_0_5])
                valid_group_5_10_hit_rate = np.mean(hit_rates[group_5_10])
                valid_group_10_15_hit_rate = np.mean(hit_rates[group_10_15])
                valid_group_15_over_hit_rate = np.mean(hit_rates[group_15_over])
            
            # # test top k recommendation
            test_mean_hit_rate = 0
            test_group_0_5_hit_rate = 0
            test_group_5_10_hit_rate = 0
            test_group_10_15_hit_rate = 0
            test_group_15_over_hit_rate = 0
            model.eval()
            with torch.no_grad():
                hit_rates = np.zeros(test_graph.num_nodes('user'))
                group_0_5 = []
                group_5_10 = []
                group_10_15 = []
                group_15_over = []
         
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
                    recommendation = rating[trained_truth_test_data[i]<1].detach().numpy()
                    recommendation = recommendation.argsort()[-TOP_K:]
                    ground_truth = np.where(ground_truth_test_data[i]>0)[0]
                    if len(ground_truth) < TOP_K:
                        hit_rates[i] = -1
                        continue

                    recommendation_set = set(recommendation)
                    ground_truth_set = set(ground_truth)

                    intersections = recommendation_set.intersection(ground_truth_set)
                    hit_rate = -1 if len(ground_truth_set) == 0 else len(intersections) / min(len(ground_truth_set), TOP_K)
                    hit_rates[i] = min(hit_rate, 1)

                    # grouping
                    if repos_per_user_test[i] < 5:
                        group_0_5.append(i)
                    elif repos_per_user_test[i] < 10:
                        group_5_10.append(i)
                    elif repos_per_user_test[i] < 15:
                        group_10_15.append(i)
                    else:
                        group_15_over.append(i)

                test_mean_hit_rate = np.mean(hit_rates[hit_rates>-1])
                test_group_0_5_hit_rate = np.mean(hit_rates[group_0_5])
                test_group_5_10_hit_rate = np.mean(hit_rates[group_5_10])
                test_group_10_15_hit_rate = np.mean(hit_rates[group_10_15])
                test_group_15_over_hit_rate = np.mean(hit_rates[group_15_over])

            print('In epoch {}, loss: {:.3f}, train_hit_rate: {:.3f}, valid hit rate: {:.3f}, test hit rate: {:.3f}'.format(
                epoch, train_avg_loss, mean_hit_rate, valid_mean_hit_rate, test_mean_hit_rate))
            # print('Valid - Group 0 to 5: {:.3f}, Group 5 to 10: {:.3f}, Group 10 to 15: {:.3f}, Group 15 to 20: {:.3f}'.format(
            #     valid_group_0_5_hit_rate, valid_group_5_10_hit_rate, valid_group_10_15_hit_rate, valid_group_15_over_hit_rate
            # ))
            # print('Test - Group 0 to 5: {:.3f}, Group 5 to 10: {:.3f}, Group 10 to 15: {:.3f}, Group 15 to 20: {:.3f}'.format(
            #     test_group_0_5_hit_rate, test_group_5_10_hit_rate, test_group_10_15_hit_rate, test_group_15_over_hit_rate
            # ))
            # print()

if __name__ == '__main__':
    t0 = time.time()
    train()
    t1 = time.time()
    running_time = t1 -t0
    print('running time: {:.3f}s'.format(running_time))