""" A collaborate filtering baseline model """
import pickle
import numpy as np
from db_manager import users, repositories, close

def similar(interaction_matrix):
    sim_user_u_v = interaction_matrix @ interaction_matrix.T
    return sim_user_u_v

def count_of_interaction(interaction_matrix, sim_user_u_v):
    top_sim = 20
    sorted_sim_u_v = np.argsort(sim_user_u_v)
    interaction_count = np.sum(interaction_matrix[sorted_sim_u_v[:, -top_sim: -1]], axis=1)
    return interaction_count

def recommend(interaction_matrix, interaction_count):
    # set already rated repositories to 0
    interaction_count[interaction_matrix>0] = 0
    # select top k to recommend
    recommendation = np.argsort(interaction_count)
    return recommendation

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
    top_k = 10
    users_count = rating_matrix.shape[0]
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
    interaction_matrix = pickle.load(open('./data/interaction_matrix.p', 'rb'))
    evaluate(interaction_matrix)