from operator import mod
import dgl
import numpy as np
from numpy.core.numeric import full
import pandas as pd
import torch
import gensim
from db_manager import users, repositories, close

class convert_to_vec:
    def __init__(self, all_users, all_repos) -> None:
        self.all_users = all_users
        self.all_repos = all_repos
        self.vector_size = 50
        self.min_count = 1
        self.epochs = 40
        self.doc_to_vec_model = gensim.models.doc2vec.Doc2Vec(vector_size=self.vector_size, min_count=self.min_count, epochs=self.epochs)
        self.train_corpus = self.get_corpus()
        self.doc_to_vec_model.build_vocab(self.train_corpus)
        self.doc_to_vec_model.train(self.train_corpus, total_examples=self.doc_to_vec_model.corpus_count, epochs=self.doc_to_vec_model.epochs)

    def get_corpus(self):
        corpus = []
        for user in self.all_users:
            corpus.append(gensim.utils.simple_preprocess(user['company']))
            corpus.append(gensim.utils.simple_preprocess(user['location']))
            corpus.append(gensim.utils.simple_preprocess(user['bio']))
        
        for repo in self.all_repos:
            corpus.append(gensim.utils.simple_preprocess(repo['name']))
            corpus.append(gensim.utils.simple_preprocess(repo['full_name']))
            corpus.append(gensim.utils.simple_preprocess(repo['description']))
            corpus.append(gensim.utils.simple_preprocess(repo['language']))
            corpus.append(gensim.utils.simple_preprocess(repo['license']))

        return corpus
    
    def infer_vector(self, sentence):
        vector = self.doc_to_vec_model.infer_vector(sentence)
        return vector

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
        'user': torch.zeros([users_count, 150], dtype=torch.float32), # 3 vectors with size 50
        'repo': torch.zeros([repos_count, 261], dtype=torch.float32) # 5 vectors with size 50 + 6 number property + 5 boolean property
    }

    # train a doc2vec model
    model = convert_to_vec(all_users, all_repos)
    vector_size = model.vector_size

    for user in all_users:
        if user['_id'] not in users_id_map:
            users_id_map[user['_id']] = user_id_counter
            user_id_counter += 1

            company = model.infer_vector(user['company'])
            location = model.infer_vector(user['location'])
            bio = model.infer_vector(user['bio'])

            graph_data['user'][current_user_id, 0:vector_size] = torch.from_numpy(company)
            graph_data['user'][current_user_id, vector_size:vector_size*2] = torch.from_numpy(location)
            graph_data['user'][current_user_id, vector_size*2:vector_size*3] = torch.from_numpy(bio)

        current_user_id = users_id_map[user['_id']]

        for repo in all_repos:
            # archived or disabled repositorues should not be considered
            if repo['archived'] is True or repo['disabled'] is True:
                continue
            if repo['_id'] not in repos_id_map:
                repos_id_map[repo['_id']] = repo_id_counter
                repo_id_counter += 1

                name = model.infer_vector(repo['name'])
                full_name = model.infer_vector(repo['full_name'])
                description = model.infer_vector(repo['description'])
                language = model.infer_vector(repo['language'])
                license = model.infer_vector(repo['license'])
                size = repo['size']
                stargazers_count = repo['stargazers_count']
                watchers_count = repo['watchers_count']
                forks_count = repo['forks_count']
                open_issues = repo['open_issues']
                subscribers_count = repo['subscribers_count']
                has_issues = 1 if repo['has_issues'] is True else 0
                has_projects = 1 if repo['has_projects'] is True else 0
                has_downloads = 1 if repo['has_downloads'] is True else 0
                has_wiki = 1 if repo['has_wiki'] is True else 0
                has_pages = 1 if repo['has_pages'] is True else 0

                graph_data['repo'][current_repo_id, 0:vector_size] = torch.from_numpy(name)
                graph_data['repo'][current_repo_id, vector_size:vector_size*2] = torch.from_numpy(full_name)
                graph_data['repo'][current_repo_id, vector_size*2:vector_size*3] = torch.from_numpy(description)
                graph_data['repo'][current_repo_id, vector_size*3:vector_size*4] = torch.from_numpy(language)
                graph_data['repo'][current_repo_id, vector_size*4:vector_size*5] = torch.from_numpy(license)
                graph_data['repo'][current_repo_id, vector_size*5:vector_size*5+1] = size
                graph_data['repo'][current_repo_id, vector_size*5:vector_size*5+2] = stargazers_count
                graph_data['repo'][current_repo_id, vector_size*5:vector_size*5+3] = watchers_count
                graph_data['repo'][current_repo_id, vector_size*5:vector_size*5+4] = forks_count
                graph_data['repo'][current_repo_id, vector_size*5:vector_size*5+5] = open_issues
                graph_data['repo'][current_repo_id, vector_size*5:vector_size*5+6] = subscribers_count
                graph_data['repo'][current_repo_id, vector_size*5:vector_size*5+7] = has_issues
                graph_data['repo'][current_repo_id, vector_size*5:vector_size*5+8] = has_projects
                graph_data['repo'][current_repo_id, vector_size*5:vector_size*5+9] = has_downloads
                graph_data['repo'][current_repo_id, vector_size*5:vector_size*5+10] = has_wiki
                graph_data['repo'][current_repo_id, vector_size*5:vector_size*5+11] = has_pages

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