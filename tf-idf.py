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

def get_file_contents(url):
    file_contents = get_json_from_url(url)
    if file_contents == -1:
        logging.error('get_file_contents got -1 when requesting from %s' % url)
        return -1
    if type(file_contents) is not list:
        logging.error('get_file_contents got wrong data structure when requesting from %s' % url)
        return -1
    return file_contents

def get_url_content(url):
    try:
        req = requests.get(url, headers=headers)
        content = req.text
    except:
        logging.error('An error occured when requesting content from this url: %s' % url)
        return -1
    if content.find('API rate limit exceeded for user') != -1:
        return check_rate_limit(get_url_content)(url)
    return content

def get_lang_ext_mappings(languages):
    exts = []
    for lang in languages.keys():
        if lang in lang_ext_mapping:
            exts += lang_ext_mapping[lang]
    return exts

def get_corpus(url, programming_languages):
    read_me_corpus = ''
    source_code_corpus = ''
    file_contents = get_file_contents(url)
    if file_contents == -1:
        return read_me_corpus, source_code_corpus
    for file_content in file_contents:
        if file_content['type'] == 'dir':
            sub_read_me_corpus, sub_source_code_corpus = get_corpus(file_content['url'], programming_languages)
            read_me_corpus += sub_read_me_corpus
            source_code_corpus += sub_source_code_corpus
        elif file_content['type'] == 'file':
            if re.match('.*\.md$', file_content['name'], re.IGNORECASE) is not None:
                content = get_url_content(file_content['download_url'])
                read_me_corpus += content
                continue
            file_ext_match = re.match('.*(\..*)$', file_content['name'], re.IGNORECASE)
            if file_ext_match is not None and file_ext_match.group(1) in programming_languages:
                content = get_url_content(file_content['download_url'])
                source_code_corpus += content
                continue
    return read_me_corpus, source_code_corpus

def get_user_repository_data():
    # select users from the database
    all_users = list(users.find({}))
    # select repos from the database
    all_repos = list(repositories.find({'$and': [{'disabled': False}, {'archived': False}]}))

    # map old user ids to new ids which start from 1
    users_id_map = {}
    user_id_counter = 0
    # map repository ids to new ids which start from 1
    repos_id_map = {}
    repo_id_counter = 0

    # user repo interactions
    interactions = np.zeros((users_count, repos_count), np.int8)
    
    read_me_tfidf = None
    read_me_corpus = []
    source_code_tfidf = None
    source_code_corpus = []
    progress_counter = 0

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
            watched_repos = user['subscriptions_id']
            if repo['_id'] in watched_repos:
                interactions[current_user_id, current_repo_id] = 1
            # star interaction
            starred_repos = user['starred_repos_id']
            if repo['_id'] in starred_repos:
                interactions[current_user_id, current_repo_id] = 2
            # fork interaction
            fork_repos = user['fork_repos_id']
            if repo['_id'] in fork_repos:
                interactions[current_user_id, current_repo_id] = 5
            # own interaction
            own_repos = user['own_repos_id']
            if repo['_id'] in own_repos:
                interactions[current_user_id, current_repo_id] = 10

            # get corpuses
            if read_me_tfidf is None or source_code_tfidf is None:
                logging.warning('Total number of repositories: %s, currently processing: %s' % (repos_count, progress_counter))
                languages = get_lang_ext_mappings(repo['languages_detail'])
                # extract the most dominant one
                languages = languages[0:1]
                read_me, source_code = get_corpus(repo['contents_url'].format(**{'+path':''}), languages)
                read_me_corpus.append(read_me)
                source_code_corpus.append(source_code)
                progress_counter += 1
        
        if read_me_tfidf is None:
            read_me_vectorizer = TfidfVectorizer()
            read_me_tfidf = read_me_vectorizer.fit_transform(read_me_corpus)
        if source_code_tfidf is None:
            source_code_vectorizer = TfidfVectorizer()
            source_code_tfidf = source_code_vectorizer.fit_transform(source_code_corpus)

        close()
        
        return interactions, read_me_tfidf, source_code_tfidf

def get_user_repo_ratings(interactions, read_me_tfidf, source_code_tfidf):
    repo_read_me_similarity = read_me_tfidf @ read_me_tfidf.T
    repo_source_code_similarity = source_code_tfidf @ source_code_tfidf.T
    alpha = 0.5
    beta = 0.5
    repo_sim = alpha * repo_read_me_similarity + beta * repo_source_code_similarity
    top_k = 2
    user_repo_ratings = np.array((users_count, repos_count))
    for i in range(users_count):
        for j in range(repos_count):
            sim_repos = repo_sim[j][interactions[i] > 0]
            top_k_sim_repos = sim_repos.argsort()[-top_k:]
            top_k_up = interactions[i][top_k_sim_repos]
            top_k_sim = sim_repos[j][top_k_sim_repos]
            user_repo_ratings[i, j] = np.dot(top_k_up, top_k_sim)
    return user_repo_ratings

def save_data(interactions, read_me_tfidf, source_code_tfidf):
    pickle.dump(interactions, open('./data/interactions.p', 'wb'))
    pickle.dump(read_me_tfidf, open('./data/read_me_tfidf.p', 'wb'))
    pickle.dump(source_code_tfidf, open('./data/source_code_tfidf.p', 'wb'))

def load_data():
    interactions = pickle.load(open('./data/interactions.p', 'rb'))
    read_me_tfidf = pickle.load(open('./data/read_me_tfidf.p', 'rb'))
    source_code_tfidf = pickle.load(open('./data/source_code_tfidf.p', 'rb'))
    return interactions, read_me_tfidf, source_code_tfidf

def evaluate(interactions, read_me_tfidf, source_code_tfidf):
    test_data = np.zeros(interactions.shape)

    for i, interaction in enumerate(interactions):
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
    
    user_repo_ratings = get_user_repo_ratings(interactions, read_me_tfidf, source_code_tfidf)
    top_k = 10
    hit_rates = np.zeros(users_count)

    for i, rating in enumerate(user_repo_ratings):
        non_test_filter = test_data[i] == 0
        rating[non_test_filter] == 0
        recommendation = rating.argsort()[-top_k:]
        ground_truth = test_data.argsort()[-top_k:]

        recommendation_set = set(recommendation)
        ground_truth_set = set(ground_truth)

        intersections = recommendation_set.intersection(ground_truth)
        hit_rate = len(intersections) / len(ground_truth_set)
        hit_rates[i] = hit_rate
    
    mean_hit_rate = np.mean(hit_rates)
    print('hit rate for top %s: %s' % (top_k, mean_hit_rate))

if __name__ == "__main__":
    if os.path.exists('./data/interactions.p') and \
        os.path.exists('./data/read_me_tfidf.p') and \
        os.path.exists('./data/source_code_tfidf.p'):
        interactions, read_me_tfidf, source_code_tfidf = load_data()
    else:
        interactions, read_me_tfidf, source_code_tfidf = get_user_repository_data()
        save_data(interactions, read_me_tfidf, source_code_tfidf)

    evaluate(interactions, read_me_tfidf, source_code_tfidf)