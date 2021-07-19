import dgl
import numpy as np
import pandas as pd
import torch
from db_manager import users, repositories, close

def get_corpus(all_users, all_repos):
    for user in all_users:
        pass

def construct_knowledge_graph():
    users_count = users.count_documents({})
    repos_count = repositories.count_documents({})

    # select users from the database
    all_users = list(users.find({}))
    # select repos from the database
    all_repos = list(repositories.find({}))

    # map old user ids to new ids which start from 1
    users_id_map = {}
    user_id_counter = 0
    # map repository ids to new ids which start from 1
    repos_id_map = {}
    repo_id_counter = 0
    # 3D table to store users-repositories interactions
    interactions = {
        'watch': pd.DataFrame(np.zeros((users_count, repos_count), np.int8)),
        'star': pd.DataFrame(np.zeros((users_count, repos_count), np.int8)),
        'fork': pd.DataFrame(np.zeros((users_count, repos_count), np.int8)),
        'own': pd.DataFrame(np.zeros((users_count, repos_count), np.int8))
    }
    # graph data
    graph_data = {
        'user': torch.zeros([users_count, 10], dtype=torch.int8),
        'repo': torch.zeros([repos_count, 20], dtype=torch.int8)
    }

    for user in all_users:
        if user['_id'] not in users_id_map:
            users_id_map[user['_id']] = user_id_counter
            user_id_counter += 1
        current_user_id = users_id_map[user['_id']]

        for repo in all_repos:
            if repo['_id'] not in repos_id_map:
                repos_id_map[repo['_id']] = repo_id_counter
                repo_id_counter += 1
            current_repo_id = repos_id_map[repo['_id']]

            # watch interaction
            watch_interaction = interactions['watch']
            watched_repos = user['subscriptions_id']
            if repo['_id'] in watched_repos:
                watch_interaction.iloc[current_user_id, current_repo_id] = 1
            # star interaction
            star_interaction = interactions['star']
            starred_repos = user['starred_repos_id']
            if repo['_id'] in starred_repos:
                star_interaction.iloc[current_user_id, current_repo_id] = 1
            # fork interaction
            fork_interaction = interactions['fork']
            fork_repos = user['fork_repos_id']
            if repo['_id'] in fork_repos:
                fork_interaction.iloc[current_user_id, current_repo_id] = 1
            # own interaction
            own_interaction = interactions['own']
            own_repos = user['own_repos_id']
            if repo['_id'] in own_repos:
                own_interaction.iloc[current_user_id, current_repo_id] = 1

        # all_repos.rewind()

    # construct source and destination nodes
    nodes_watch = np.where(interactions['watch'] == 1)
    nodes_star = np.where(interactions['star'] == 1)
    nodes_fork = np.where(interactions['fork'] == 1)
    nodes_own = np.where(interactions['own'] == 1)

    num_nodes_dict = { 'user': users_count, 'repo': repos_count }

    # construct the heterograph from the dataframe
    g = dgl.heterograph({
        ('user', 'star', 'repo'): (nodes_star[0], nodes_star[1]),
        ('user', 'watch', 'repo'): (nodes_watch[0], nodes_watch[1]),
        ('user', 'fork', 'repo'): (nodes_fork[0], nodes_fork[1]),
        ('user', 'own', 'repo'): (nodes_own[0], nodes_own[1])
    }, num_nodes_dict=num_nodes_dict, device='cuda')
    # set the node data
    g.ndata['graph_data'] = graph_data

    close()

    return g

if __name__ == "__main__":
    construct_knowledge_graph()