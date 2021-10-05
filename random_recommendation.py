import numpy as np
import pickle

def top_k_evaluate(top_k, rating_matrix, train_data, test_data):
    users_count = rating_matrix.shape[0]
    repos_count = rating_matrix.shape[1]
    hit_rates = np.zeros(users_count)
    group_0_5 = []
    group_5_10 = []
    group_10_15 = []
    group_15_over = []

    for i, rating in enumerate(rating_matrix):
        non_train_mask = np.invert(train_data[i] > 0)
        recommendation = np.random.choice(np.where(non_train_mask)[0], size=top_k, replace=False)
        ground_truth = np.where(test_data[i]>0)[0]

        intersections = np.intersect1d(recommendation, ground_truth)
        hit_rate = -1 if len(ground_truth) == 0 else len(intersections) / min(len(ground_truth), top_k)
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

    return (
        mean_hit_rate,
        group_0_5_hit_rate,
        group_5_10_hit_rate,
        group_10_15_hit_rate,
        group_15_over_hit_rate
    )

def evaluate(rating_matrix):
    train_data = np.zeros(rating_matrix.shape)
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
        train_data[i][up_index[train_mask]] = interaction[up_index[train_mask]]
    
    top_k = [10, 15, 20]
    training_results = []

    for k in top_k:
        (
            mean_hit_rate,
            group_0_5_hit_rate,
            group_5_10_hit_rate,
            group_10_15_hit_rate,
            group_15_over_hit_rate
        ) = top_k_evaluate(k, rating_matrix, train_data, test_data)

        training_results.append([
            k, 
            mean_hit_rate,
            group_0_5_hit_rate,
            group_5_10_hit_rate,
            group_10_15_hit_rate,
            group_15_over_hit_rate
        ])

    for result in training_results:
        print('hit rate for top %s: %.3f, Group 0 to 5: %.3f, Group 5 to 10: %.3f, Group 10 to 15: %.3f, Group 15 to 20: %.3f' % (
            result[0], 
            result[1],
            result[2],
            result[3],
            result[4],
            result[5]
        ))

if __name__ == "__main__":
    interaction_matrix = pickle.load(open('./data/interaction_matrix.p', 'rb'))
    evaluate(interaction_matrix)