import config as cfg
import requests
import logging
import re
from logging.handlers import RotatingFileHandler
import pandas as pd
import numpy as np
import pickle
import os
from lang_ext_mapping import lang_ext_mapping
from db_manager import users, repositories, close
from data_collection import get_json_from_url, check_rate_limit, logger
from sklearn.feature_extraction.text import TfidfVectorizer

headers = {'Authorization': 'token %s' % cfg.oauth_token}

users_count = users.count_documents({})
repos_count = repositories.count_documents({'$and': [{'disabled': False}, {'archived': False}]})

def read_me_corpus():
    all_repos = repositories.find({'$and': [{'disabled': False}, {'archived': False}]})
    for repo in all_repos:
        read_me_corpus = ''
        if 'read_me' in repo:
            for file_name, file_content in repo['read_me'].items():
                read_me_corpus += file_content
        yield read_me_corpus
   
def source_code_corpus():
    all_repos = repositories.find({'$and': [{'disabled': False}, {'archived': False}]})
    for repo in all_repos:
        source_code_corpus = ''
        if 'source_code' in repo:
            for file_name, file_content in repo['source_code'].items():
                source_code_corpus += file_content
        yield source_code_corpus

def get_read_me_tfidf():
    read_me_vectorizer = TfidfVectorizer()
    read_me_tfidf = read_me_vectorizer.fit_transform(read_me_corpus())
    return read_me_tfidf

def get_source_code_tfidf():
    source_code_vectorizer = TfidfVectorizer()
    source_code_tfidf = source_code_vectorizer.fit_transform(source_code_corpus())
    return source_code_tfidf

def get_rating_matrix():
    users_id_map = pickle.load(open('./data/users_id_map.p', 'rb'))
    repos_id_map = pickle.load(open('./data/repos_id_map.p', 'rb'))
    interaction_matrix = pickle.load(open('./data/interaction_matrix.p', 'rb'))

    # select users from the database
    all_users = list(users.find({}))

    # user repo interactions
    rating_matrix = np.zeros((users_count, repos_count), np.int8)

    for user in all_users:
        current_user_id = users_id_map[user['_id']]

        for repo_id, repo_index in repos_id_map.items():
            current_repo_id = repo_index

            if interaction_matrix[current_user_id, current_repo_id] == 0:
                continue

            # watch interaction
            watched_repos = user['subscriptions_id']
            if repo_id in watched_repos:
                rating_matrix[current_user_id, current_repo_id] = 1
            # star interaction
            starred_repos = user['starred_repos_id']
            if repo_id in starred_repos:
                rating_matrix[current_user_id, current_repo_id] = 2
            # fork interaction
            fork_repos = user['fork_repos_id']
            if repo_id in fork_repos:
                rating_matrix[current_user_id, current_repo_id] = 5
            # own interaction
            own_repos = user['own_repos_id']
            if repo_id in own_repos:
                rating_matrix[current_user_id, current_repo_id] = 10
    
    return rating_matrix

def get_user_repo_ratings(rating_matrix, read_me_tfidf, source_code_tfidf):
    repo_read_me_similarity = read_me_tfidf @ read_me_tfidf.T
    repo_source_code_similarity = source_code_tfidf @ source_code_tfidf.T
    alpha = 0.5
    beta = 0.5
    repo_sim = alpha * repo_read_me_similarity + beta * repo_source_code_similarity
    repo_sim = repo_sim.toarray()
    top_k = 2
    user_repo_ratings = np.zeros((users_count, repos_count))
    for i in range(users_count):
        for j in range(repos_count):
            sim_repos = repo_sim[j][rating_matrix[i] > 0]
            top_k_sim_repos = sim_repos.argsort()[-top_k:]
            top_k_up = rating_matrix[i][top_k_sim_repos]
            top_k_sim = sim_repos[top_k_sim_repos]
            user_repo_ratings[i, j] = np.dot(top_k_up, top_k_sim)
    return user_repo_ratings

def evaluate(rating_matrix, read_me_tfidf, source_code_tfidf):
    test_data = np.zeros(rating_matrix.shape)

    for i, interaction in enumerate(rating_matrix):
        up_index = np.where(interaction > 0)[0]
        interaction_count = len(up_index)

        # sample data for train and test with the ratio of 60:40
        train_number = round(interaction_count * 0.6)
        test_number = interaction_count- train_number
        sample_indexes = np.array([0] * train_number + [1] * test_number)
        np.random.shuffle(sample_indexes)

        # generate the train and test mask
        train_mask = sample_indexes==0
        test_mask = sample_indexes==1

        test_data[i][up_index[test_mask]] = interaction[up_index[test_mask]]
        # set test data to 0
        interaction[up_index[test_mask]] = 0
    
    user_repo_ratings = get_user_repo_ratings(rating_matrix, read_me_tfidf, source_code_tfidf)
    top_k = 10
    hit_rates = np.zeros(users_count)

    for i, rating in enumerate(user_repo_ratings):
        non_test_filter = test_data[i] == 0
        rating[non_test_filter] == 0
        recommendation = rating.argsort()[-top_k:]
        ground_truth = test_data[i].argsort()[-top_k:]

        recommendation_set = set(recommendation)
        ground_truth_set = set(ground_truth)

        intersections = recommendation_set.intersection(ground_truth)
        hit_rate = 0 if len(ground_truth_set) == 0 else len(intersections) / min(len(ground_truth_set), top_k)
        hit_rates[i] = hit_rate
    
    mean_hit_rate = np.mean(hit_rates)
    print('hit rate for top %s: %s' % (top_k, mean_hit_rate))

if __name__ == "__main__":
    if os.path.exists('./data/rating_matrix.p'):
        rating_matrix = pickle.load(open('./data/rating_matrix.p', 'rb'))
    else:
        rating_matrix = get_rating_matrix()
        pickle.dump(rating_matrix, open('./data/rating_matrix.p', 'wb'))
    if os.path.exists('./data/read_me_tfidf.p'):
        read_me_tfidf = pickle.load(open('./data/read_me_tfidf.p', 'rb'))
    else:
        read_me_tfidf = get_read_me_tfidf()
        pickle.dump(read_me_tfidf, open('./data/read_me_tfidf.p', 'wb'))
    if os.path.exists('./data/source_code_tfidf.p'):
        source_code_tfidf = pickle.load(open('./data/source_code_tfidf.p', 'rb'))
    else:
        source_code_tfidf = get_source_code_tfidf()
        pickle.dump(source_code_tfidf, open('./data/source_code_tfidf.p', 'wb'))

    # close the database
    close()

    evaluate(rating_matrix, read_me_tfidf, source_code_tfidf)