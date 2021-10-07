""" A collaborate filtering baseline model """
import pickle
import numpy as np
from util import Group

def similar(interaction_matrix):
    sim_user_u_v = interaction_matrix @ interaction_matrix.T
    return sim_user_u_v

def count_of_interaction(interaction_matrix, sim_user_u_v):
    top_sim = 20
    sorted_sim_u_v = np.argsort(sim_user_u_v)
    interaction_count = np.sum(interaction_matrix[sorted_sim_u_v[:, -top_sim - 1: -1]], axis=1)
    return interaction_count

def recommend(interaction_matrix, interaction_count):
    # set already rated repositories to 0
    interaction_count[interaction_matrix>0] = 0
    # select top k to recommend00
    recommendation = np.argsort(interaction_count)
    return recommendation

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

    for i, rating in enumerate(user_repo_ratings):
        recommendation = rating[-top_k:]
        index_argsorted = test_data[i].argsort()
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
        else:
            mrr[i] == -1

        # grouping
        repos_count = len(test_data[i][test_data[i]>0])
        if repos_count < 5:
            hit_rate_groups['0-5'].append(i)
            mrr_groups['0-5'].append(i)
        elif repos_count < 10:
            hit_rate_groups['5-10'].append(i)
            mrr_groups['5-10'].append(i)
        elif repos_count < 15:
            hit_rate_groups['10-15'].append(i)
            mrr_groups['10-15'].append(i)
        else:
            hit_rate_groups['15-over'].append(i)
            mrr_groups['15-over'].append(i)
    
    # hit rate mean
    mean_hit_rate = np.mean(hit_rates[hit_rates>-1])
    for group_name, group_indices in hit_rate_groups.items():
        group_hit_rate[group_name] = np.mean(hit_rates[group_indices][hit_rates[group_indices]>-1])

    # mrr mean
    mean_mrr = np.mean(mrr[mrr>-1])
    for group_name, group_indices in mrr_groups.items():
        group_mrr[group_name] = np.mean(mrr[group_indices][mrr[group_indices]>-1])

    return (
        mean_hit_rate,
        mean_mrr,
        group_hit_rate,
        group_mrr
    )

def evaluate(rating_matrix):
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
    
    similarities = similar(rating_matrix)
    interaction_count = count_of_interaction(rating_matrix, similarities)
    user_repo_ratings = recommend(rating_matrix, interaction_count)
    top_k = [10, 15, 20]
    training_results = []
    result_title_str = 'top %s, hit rate: %.3f, MRR: %.3f'
    group_title_str = ''

    for k in top_k:
        (
            mean_hit_rate,
            mean_mrr,
            group_hit_rate,
            group_mrr
        ) = top_k_evaluate(k, rating_matrix, user_repo_ratings, test_data)

        result = [
            k,
            mean_hit_rate,
            mean_mrr
        ]

        for name, value in group_hit_rate.items():
            result.append(value)
            if k == top_k[0]:
                group_title_str += ', Hit rate group ' + name + ': %.3f'

        for name, value in group_mrr.items():
            result.append(value)
            if k == top_k[0]:
                group_title_str += ', MRR group ' + name + ': %.3f'

        training_results.append(result)

    for result in training_results:
        print((result_title_str + group_title_str) % tuple(result))

if __name__ == "__main__":
    interaction_matrix = pickle.load(open('./data/interaction_matrix.p', 'rb'))
    evaluate(interaction_matrix)