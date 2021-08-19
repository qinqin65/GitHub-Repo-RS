import dgl
import numpy as np
import pandas as pd
import pickle
from db_manager import users, repositories, close

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
    # interactions map
    interactions_map = {
        'watch': 0,
        'star': 1,
        'fork': 2,
        'own': 3
    }
    # 3D table to store users-repositories interactions
    interactions = (
        pd.DataFrame(np.zeros((users_count, repos_count), np.int8)),
        pd.DataFrame(np.zeros((users_count, repos_count), np.int8)),
        pd.DataFrame(np.zeros((users_count, repos_count), np.int8)),
        pd.DataFrame(np.zeros((users_count, repos_count), np.int8))
    )

    for user in all_users:
        if user['_id'] not in users_id_map:
            users_id_map[user['_id']] = user_id_counter
            user_id_counter += 1

        current_user_id = users_id_map[user['_id']]

        for repo in all_repos:
            # archived or disabled repositorues should not be considered
            if repo['archived'] is True or repo['disabled'] is True:
                continue
            if repo['_id'] not in repos_id_map:
                repos_id_map[repo['_id']] = repo_id_counter
                repo_id_counter += 1

            current_repo_id = repos_id_map[repo['_id']]

            # watch interaction
            watch_interaction = interactions[interactions_map['watch']]
            watched_repos = user['subscriptions_id']
            if repo['_id'] in watched_repos:
                watch_interaction.iloc[current_user_id, current_repo_id] = 1
            # star interaction
            star_interaction = interactions[interactions_map['star']]
            starred_repos = user['starred_repos_id']
            if repo['_id'] in starred_repos:
                star_interaction.iloc[current_user_id, current_repo_id] = 1
            # fork interaction
            fork_interaction = interactions[interactions_map['fork']]
            fork_repos = user['fork_repos_id']
            if repo['_id'] in fork_repos:
                fork_interaction.iloc[current_user_id, current_repo_id] = 1
            # own interaction
            own_interaction = interactions[interactions_map['own']]
            own_repos = user['own_repos_id']
            if repo['_id'] in own_repos:
                own_interaction.iloc[current_user_id, current_repo_id] = 1

    # construct source and destination nodes
    nodes_watch = np.where(interactions[interactions_map['watch']] == 1)
    nodes_star = np.where(interactions[interactions_map['star']] == 1)
    nodes_fork = np.where(interactions[interactions_map['fork']] == 1)
    nodes_own = np.where(interactions[interactions_map['own']] == 1)

    num_nodes_dict = { 'user': users_count, 'repo': repos_count }

    # construct the heterograph from the dataframe
    g = dgl.heterograph({
        ('user', 'star', 'repo'): (nodes_star[0], nodes_star[1]),
        ('user', 'watch', 'repo'): (nodes_watch[0], nodes_watch[1]),
        ('user', 'fork', 'repo'): (nodes_fork[0], nodes_fork[1]),
        ('user', 'own', 'repo'): (nodes_own[0], nodes_own[1])
    }, num_nodes_dict=num_nodes_dict, device='cuda')

    close()

    return g, users_id_map, repos_id_map, interactions_map

if __name__ == "__main__":
    knowledge_graph, users_id_map, repos_id_map, interactions_map = construct_knowledge_graph()
    dgl.save_graphs('./data/saved_kowledge_graph.bin', [knowledge_graph])
    pickle.dump(users_id_map, open('./data/users_id_map.p', 'wb'))
    pickle.dump(repos_id_map, open('./data/repos_id_map.p', 'wb'))
    pickle.dump(interactions_map, open('./data/interactions_map.p', 'wb'))