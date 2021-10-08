import config as cfg
import numpy as np
import pickle
import os
from db_manager import users, repositories, close
from sklearn.feature_extraction.text import TfidfVectorizer
from util import Group

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

def top_k_evaluate(top_k, rating_matrix, user_repo_ratings, test_data):
    users_count = rating_matrix.shape[0]
    # hit rate
    hit_rates = np.zeros(users_count)
    group_hit_rate = {}
    hit_rate_groups = Group()

    # MRR
    mrr = np.zeros(users_count)
    group_mrr = {}
    mrr_groups = Group()

    # nDCG
    ndcg = np.zeros(users_count)
    group_ndcg = {}
    ndcg_groups = Group()

    for i, rating in enumerate(user_repo_ratings):
        recommendation = rating.argsort()[::-1][:top_k]
        index_argsorted = test_data[i].argsort()[::-1]
        filter_index = test_data[i][index_argsorted] > 0
        ground_truth = index_argsorted[filter_index]

        intersections, recommendation_index, ground_truth_index = np.intersect1d(recommendation, ground_truth, return_indices=True)
        number_of_ground_truth = len(ground_truth)
        number_of_intersections = len(intersections)

        # hit rate
        hit_rate = -1 if number_of_ground_truth == 0 else number_of_intersections / min(number_of_ground_truth, top_k)
        hit_rates[i] = hit_rate

        # MRR
        if number_of_intersections > 0:
            if recommendation_index[0] <= top_k:
                mrr[i] = 1 / (recommendation_index[0] + 1)
        elif number_of_ground_truth == 0:
            mrr[i] == -1
        
        # nDCG
        if number_of_ground_truth == 0:
            ndcg[i] = -1
        else:
            relevance_score = test_data[i][recommendation][:min(number_of_ground_truth, top_k)]
            relevance_score_idea = test_data[i][ground_truth][:min(number_of_ground_truth, top_k)]
            pow_rel = np.power(2, relevance_score) - 1
            pow_rel_idea = np.power(2, relevance_score_idea) - 1
            ranks = np.arange(start=1, stop=len(relevance_score) + 1)
            log_rank = np.log2(ranks + 1)
            dcg = np.sum(pow_rel / log_rank)
            idcg = np.sum(pow_rel_idea / log_rank)
            ndcg[i] = dcg / idcg
    
        # grouping
        repos_count = len(test_data[i][test_data[i]>0])
        if repos_count < 5:
            hit_rate_groups['0-5'].append(i)
            mrr_groups['0-5'].append(i)
            ndcg_groups['0-5'].append(i)
        elif repos_count < 10:
            hit_rate_groups['5-10'].append(i)
            mrr_groups['5-10'].append(i)
            ndcg_groups['5-10'].append(i)
        elif repos_count < 15:
            hit_rate_groups['10-15'].append(i)
            mrr_groups['10-15'].append(i)
            ndcg_groups['10-15'].append(i)
        else:
            hit_rate_groups['15-over'].append(i)
            mrr_groups['15-over'].append(i)
            ndcg_groups['15-over'].append(i)

    # hit rate mean
    mean_hit_rate = np.mean(hit_rates[hit_rates>-1])
    for group_name, group_indices in hit_rate_groups.items():
        group_hit_rate[group_name] = np.mean(hit_rates[group_indices][hit_rates[group_indices]>-1])

    # mrr mean
    mean_mrr = np.mean(mrr[mrr>-1])
    for group_name, group_indices in mrr_groups.items():
        group_mrr[group_name] = np.mean(mrr[group_indices][mrr[group_indices]>-1])

    # nDCG mean
    mean_ndcg = np.mean(ndcg[ndcg>-1])
    for group_name, group_indices in ndcg_groups.items():
        group_ndcg[group_name] = np.mean(ndcg[group_indices][ndcg[group_indices]>-1])

    return (
        mean_hit_rate,
        mean_mrr,
        mean_ndcg,
        group_hit_rate,
        group_mrr,
        group_ndcg
    )

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
    top_k = [10, 15, 20]
    training_results = []
    result_title_str = 'top %s, hit rate: %.3f, MRR: %.3f, nDCG: %.3f'
    group_title_str = ''

    for k in top_k:
        (
            mean_hit_rate,
            mean_mrr,
            mean_ndcg,
            group_hit_rate,
            group_mrr,
            group_ndcg
        ) = top_k_evaluate(k, rating_matrix, user_repo_ratings, test_data)

        result = [
            k,
            mean_hit_rate,
            mean_mrr,
            mean_ndcg
        ]

        for name, value in group_hit_rate.items():
            result.append(value)
            if k == top_k[0]:
                group_title_str += ', Hit rate group ' + name + ': %.3f'

        for name, value in group_mrr.items():
            result.append(value)
            if k == top_k[0]:
                group_title_str += ', MRR group ' + name + ': %.3f'
        
        for name, value in group_ndcg.items():
            result.append(value)
            if k == top_k[0]:
                group_title_str += ', nDCG group ' + name + ': %.3f'

        training_results.append(result)

    for result in training_results:
        print((result_title_str + group_title_str) % tuple(result))

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