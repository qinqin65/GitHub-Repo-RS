import dgl
import torch
import numpy as np
import time
import pickle
from KGCN import Model, loss_fn
from sklearn.metrics import roc_auc_score
from util import Group

def process_graph(graph: dgl.heterograph):
    number_of_users = graph.num_nodes('user')
    number_of_repos = graph.num_nodes('repo')
    ground_truth = torch.zeros((number_of_users, number_of_repos), dtype=torch.int8)
    user_repo = torch.zeros((number_of_users, number_of_repos), dtype=torch.int8)
    interactions_map = pickle.load(open('./data/interactions_map.p', 'rb'))
    for etype in graph.canonical_etypes:
        # ignore the reverse relation
        if etype[0] != 'user' and etype[1] != 'repo':
            continue

        edges = graph.edges(etype=etype)

        ground_truth[edges[0], edges[1]] = interactions_map[etype[1]] + 1
        user_repo[edges[0], edges[1]] = 1
    
    repos_per_user = torch.sum(user_repo, axis=1)

    return ground_truth, repos_per_user.numpy()

def process_edge_data(graph, edge_data):
    number_of_users = graph.num_nodes('user')
    number_of_repos = graph.num_nodes('repo')
    ratings = np.zeros((number_of_users, number_of_repos))
    for etype in graph.canonical_etypes:
        # ignore the reverse relation
        if etype[0] != 'user' and etype[1] != 'repo':
            continue
        
        edges = graph.edges(etype=etype)
        data = torch.squeeze(edge_data[etype]).numpy()
        ratings[edges[0], edges[1]] = np.maximum(ratings[edges[0], edges[1]], data)
    
    return ratings

def compute_auc(graph, pos_score, neg_score):
    auc_scores = []
    for etype in graph.canonical_etypes:
        # ignore the reverse relation
        if etype[0] != 'user' and etype[1] != 'repo':
            continue
        
        pos_data = torch.squeeze(pos_score[etype])
        neg_data = torch.squeeze(neg_score[etype])
        
        scores = torch.cat([pos_data, neg_data]).numpy()
        labels = torch.cat(
            [torch.ones(pos_data.shape[0]), torch.zeros(neg_data.shape[0])]).numpy()

        auc_score = roc_auc_score(labels, scores)

        auc_scores.append(auc_score)
    
    return np.mean(auc_scores)

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
    train_pos_g = g[3]
    train_neg_g = g[4]
    valid_pos_g = g[5]
    valid_neg_g = g[6]
    test_pos_g = g[7]
    test_neg_g = g[8]
    
    model = Model(train_graph, USER_INPUT_SIZE, REPO_INPUT_SIZE, USER_REPO_OUTPUT_SIZE, HIDDEN_OUTPUT_SIZE, OUT_SIZE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    ground_truth_valid_data, repos_per_user_valid = process_graph(valid_graph)
    ground_truth_test_data, repos_per_user_test = process_graph(test_graph)

    for epoch in range(EPOCH):
        training_loops = 0
        total_loss = 0

        user_feat = train_graph.ndata['graph_data']['user']
        repo_feat = train_graph.ndata['graph_data']['repo']
        model.train()
        pos_score, neg_score = model(train_graph, train_pos_g, train_neg_g, user_feat, repo_feat)
        loss = loss_fn(pos_score, neg_score, NEG_SAMPLE_SIZE)

        total_loss += loss.item()
        training_loops += 1

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_avg_loss = total_loss / training_loops

        if epoch % 5 == 0:
            # valid top k recommendation
            model.eval()
            with torch.no_grad():
                # hit rate
                valid_mean_hit_rate = 0
                valid_group_hit_rate = {}
                hit_rates = np.zeros(valid_graph.num_nodes('user'))
                hit_rate_groups = Group()

                # MRR
                valid_mrr = 0
                valid_group_mrr = {}
                mrr = np.zeros(valid_graph.num_nodes('user'))
                mrr_groups = Group()
         
                h_user = model.user_embedding(train_graph.ndata['graph_data']['user'])
                h_repo = model.repo_embedding(train_graph.ndata['graph_data']['repo'])

                h_dict = {
                    'user': h_user,
                    'repo': h_repo
                }

                h = model.hidden(train_graph, h_dict)
                out = model.out(train_graph, h)

                prediction = model.predict(valid_graph, out)
                user_repo_rating = process_edge_data(valid_graph, prediction)

                for i, rating in enumerate(user_repo_rating):
                    recommendation = rating.argsort()[::-1][:TOP_K]
                    index_sorted = ground_truth_valid_data[i].argsort(descending=True)
                    filter_index = ground_truth_valid_data[i][index_sorted] > 0
                    ground_truth = index_sorted[filter_index]

                    intersections, recommendation_index, ground_truth_index = np.intersect1d(recommendation, ground_truth, return_indices=True)
                    number_of_intersections = len(intersections)
                    number_of_ground_truth = len(ground_truth)

                    # hit rate
                    hit_rate = -1 if number_of_ground_truth == 0 else number_of_intersections / min(number_of_ground_truth, TOP_K)
                    hit_rates[i] = min(hit_rate, 1)

                    # MRR
                    if number_of_intersections > 0:
                        if recommendation_index[0] <= TOP_K:
                            mrr[i] = 1 / (recommendation_index[0] + 1)
                    else:
                        mrr[i] == -1

                    # grouping
                    if repos_per_user_valid[i] < 5:
                        hit_rate_groups['0-5'].append(i)
                        mrr_groups['0-5'].append(i)
                    elif repos_per_user_valid[i] < 10:
                        hit_rate_groups['5-10'].append(i)
                        mrr_groups['5-10'].append(i)
                    elif repos_per_user_valid[i] < 15:
                        hit_rate_groups['10-15'].append(i)
                        mrr_groups['10-15'].append(i)
                    else:
                        hit_rate_groups['15-over'].append(i)
                        mrr_groups['15-over'].append(i)

                # hit rate mean
                valid_mean_hit_rate = np.mean(hit_rates[hit_rates>-1])
                for group_name, group_indices in hit_rate_groups.items():
                    valid_group_hit_rate[group_name] = np.mean(hit_rates[group_indices][hit_rates[group_indices]>-1])

                # mrr mean
                valid_mrr = np.mean(mrr[mrr>-1])
                for group_name, group_indices in mrr_groups.items():
                    valid_group_mrr[group_name] = np.mean(mrr[group_indices][mrr[group_indices]>-1])
            
            # test top k recommendation
            model.eval()
            with torch.no_grad():
                # hit rate
                test_mean_hit_rate = 0
                test_group_hit_rate = {}
                hit_rates = np.zeros(test_graph.num_nodes('user'))
                hit_rate_groups = Group()

                # MRR
                test_mrr = 0
                test_group_mrr = {}
                mrr = np.zeros(test_graph.num_nodes('user'))
                mrr_groups = Group()
         
                h_user = model.user_embedding(train_graph.ndata['graph_data']['user'])
                h_repo = model.repo_embedding(train_graph.ndata['graph_data']['repo'])

                h_dict = {
                    'user': h_user,
                    'repo': h_repo
                }

                h = model.hidden(train_graph, h_dict)
                out = model.out(train_graph, h)

                prediction = model.predict(test_graph, out)
                user_repo_rating = process_edge_data(test_graph, prediction)

                for i, rating in enumerate(user_repo_rating):
                    recommendation = rating.argsort()[::-1][:TOP_K]
                    index_sorted = ground_truth_test_data[i].argsort(descending=True)
                    filter_index = ground_truth_test_data[i][index_sorted] > 0
                    ground_truth = index_sorted[filter_index]

                    intersections, recommendation_index, ground_truth_index = np.intersect1d(recommendation, ground_truth, return_indices=True)
                    number_of_intersections = len(intersections)
                    number_of_ground_truth = len(ground_truth)

                    # hit rate
                    hit_rate = -1 if number_of_ground_truth == 0 else number_of_intersections / min(number_of_ground_truth, TOP_K)
                    hit_rates[i] = hit_rate

                    # MRR
                    if number_of_intersections > 0:
                        if recommendation_index[0] <= TOP_K:
                            mrr[i] = 1 / (recommendation_index[0] + 1)
                    else:
                        mrr[i] == -1

                    # grouping
                    if repos_per_user_test[i] < 5:
                        hit_rate_groups['0-5'].append(i)
                        mrr_groups['0-5'].append(i)
                    elif repos_per_user_test[i] < 10:
                        hit_rate_groups['5-10'].append(i)
                        mrr_groups['5-10'].append(i)
                    elif repos_per_user_test[i] < 15:
                        hit_rate_groups['10-15'].append(i)
                        mrr_groups['10-15'].append(i)
                    else:
                        hit_rate_groups['15-over'].append(i)
                        mrr_groups['15-over'].append(i)

                # hit rate mean
                test_mean_hit_rate = np.mean(hit_rates[hit_rates>-1])
                for group_name, group_indices in hit_rate_groups.items():
                    test_group_hit_rate[group_name] = np.mean(hit_rates[group_indices][hit_rates[group_indices]>-1])

                # mrr mean
                test_mrr = np.mean(mrr[mrr>-1])
                for group_name, group_indices in mrr_groups.items():
                    test_group_mrr[group_name] = np.mean(mrr[group_indices][mrr[group_indices]>-1])
            
            # conpute the AUC score
            auc_score = 0
            model.eval()
            with torch.no_grad():
                h_user = model.user_embedding(train_graph.ndata['graph_data']['user'])
                h_repo = model.repo_embedding(train_graph.ndata['graph_data']['repo'])

                h_dict = {
                    'user': h_user,
                    'repo': h_repo
                }

                h = model.hidden(train_graph, h_dict)
                out = model.out(train_graph, h)

                pos_score = model.predict(test_pos_g, out)
                neg_score = model.predict(test_neg_g, out)

                auc_score = compute_auc(train_graph, pos_score, neg_score)

            print('In epoch {}, loss: {:.3f}, auc: {:.3f}, valid hit rate: {:.3f}, test hit rate: {:.3f}, valid MRR: {:.3f}, test MRR: {:.3f}'.format(
                epoch, train_avg_loss, auc_score, valid_mean_hit_rate, test_mean_hit_rate, valid_mrr, test_mrr))
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