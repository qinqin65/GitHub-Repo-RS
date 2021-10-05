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
    alpha = 0.9
    beta = 0.1
    repo_sim = alpha * repo_read_me_similarity + beta * repo_source_code_similarity
    repo_sim = repo_sim.toarray()
    top_k = 2
    user_repo_ratings = np.zeros((users_count, repos_count))
    for i in range(users_count):
        for j in range(repos_count):
            # eliminate the current repositories itslef
            repo_sim[j][j] = 0
            # the reposittories the user rated
            user_repos = np.where(rating_matrix[i]>0)[0]
            # the similarity of the user rated repositories
            similarities = repo_sim[j][user_repos]
            similarities_arg_sorted = similarities.argsort()
            # select top k similarities
            similarities_arg_sorted = similarities_arg_sorted[-top_k:]
       
            top_k_up = rating_matrix[i][user_repos[similarities_arg_sorted]]
            top_k_sim = similarities[similarities_arg_sorted]
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
    group_0_5 = []
    group_5_10 = []
    group_10_15 = []
    group_15_over = []

    for i, rating in enumerate(user_repo_ratings):
        recommendation = rating.argsort()[-top_k:]
        ground_truth = np.where(test_data[i]>0)[0]

        intersections = np.intersect1d(recommendation, ground_truth)
        number_of_ground_truth = len(ground_truth)
        number_of_intersections = len(intersections)
        hit_rate = -1 if number_of_ground_truth == 0 else number_of_intersections / min(number_of_ground_truth, top_k)
        hit_rates[i] = hit_rate
    
        # grouping
        repos_count = len(test_data[i][test_data[i]>0])
        if repos_count < 5:
            group_0_5.append(i)
        elif repos_count < 10:
            group_5_10.append(i)
        elif repos_count < 15:
            group_10_15.append(i)
        else:
            group_15_over.append(i)

    mean_hit_rate = np.mean(hit_rates[hit_rates>-1])
    group_0_5_hit_rate = np.mean(hit_rates[group_0_5][hit_rates[group_0_5]>-1])
    group_5_10_hit_rate = np.mean(hit_rates[group_5_10][hit_rates[group_5_10]>-1])
    group_10_15_hit_rate = np.mean(hit_rates[group_10_15][hit_rates[group_10_15]>-1])
    group_15_over_hit_rate = np.mean(hit_rates[group_15_over][hit_rates[group_15_over]>-1])
    print('hit rate for top %s: %.3f, Group 0 to 5: %.3f, Group 5 to 10: %.3f, Group 10 to 15: %.3f, Group 15 to 20: %.3f' % (
        top_k, 
        mean_hit_rate,
        group_0_5_hit_rate,
        group_5_10_hit_rate,
        group_10_15_hit_rate,
        group_15_over_hit_rate
    ))

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