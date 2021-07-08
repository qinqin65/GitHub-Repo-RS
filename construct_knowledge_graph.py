import dgl
from dgl.convert import graph
import pymongo
from tensorflow.python.eager.context import device
import config as cfg
import numpy as np
import pandas as pd
import tensorflow as tf

# No Sql database which will be used to store the users and projects data
client = pymongo.MongoClient(cfg.db_conn_str, serverSelectionTimeoutMS=5000)
db = client.github_project
users = db.users
repositories = db.repositories

# map old user ids to new ids which start from 1
users_id_map = {}
user_id_counter = 0
# map repository ids to new ids which start from 1
repos_id_map = {}
repo_id_counter = 0
# 3D table to store users-repositories interactions
users_count = users.count_documents({})
repos_count = repositories.count_documents({})
interactions = {
    'watch': pd.DataFrame(np.zeros((users_count, repos_count))),
    'star': pd.DataFrame(np.zeros((users_count, repos_count))),
    'fork': pd.DataFrame(np.zeros((users_count, repos_count))),
    'own': pd.DataFrame(np.zeros((users_count, repos_count)))
}

# select users from the database
all_users = users.find({})
# select repos from the database
all_repos = repositories.find({})
# graph data
graph_data = {
    'user': tf.zeros([users_count, 10]),
    'repo': tf.zeros([repos_count, 20])
}

for user in all_users:
    users_id_map[user['_id']] = user_id_counter
    current_user_id = users_id_map[user['_id']]
    user_id_counter += 1

    for repo in all_repos:
        repos_id_map[repo['_id']] = repo_id_counter
        current_repo_id = repos_id_map[repo['_id']]
        repo_id_counter += 1

        # watch interaction
        watch_interaction = interactions['watch']
        watched_repos = user['subscriptions']
        if repo['_id'] in watched_repos:
            watch_interaction.iloc[current_user_id, current_repo_id] = 1
        # star interaction
        star_interaction = interactions['star']
        starred_repos = user['starred_repos']
        if repo['_id'] in starred_repos:
            star_interaction.iloc[current_user_id, current_repo_id] = 1
        # fork interaction
        fork_interaction = interactions['fork']
        # own interaction
        own_interaction = interactions['own']
        user_repos = user['repos']
        if repo['_id'] in user_repos:
            if repo['fork'] is True:
                fork_interaction.iloc[current_user_id, current_repo_id] = 1
            else:
                own_interaction.iloc[current_user_id, current_repo_id] = 1

# construct source and destination nodes
dest_src_nodes_watch = np.where(interactions['watch'] == 1)
dest_src_nodes_star = np.where(interactions['star'] == 1)
dest_src_nodes_fork = np.where(interactions['fork'] == 1)
dest_src_nodes_own = np.where(interactions['own'] == 1)

# construct the heterograph from the dataframe
g = dgl.heterograph({
    ('user', 'watch', 'repo'): (dest_src_nodes_watch[1], dest_src_nodes_watch[0]),
    ('user', 'star', 'repo'): (dest_src_nodes_star[1], dest_src_nodes_star[0]),
    ('user', 'fork', 'repo'): (dest_src_nodes_fork[1], dest_src_nodes_fork[0]),
    ('user', 'own', 'repo'): (dest_src_nodes_own[1], dest_src_nodes_own[0])
}, device='gpu:0')
# set the node data
g.ndata = graph_data