import argparse
import os
import pandas as pd
from statistics import mean
from typing import Dict

from pandas import Series

from user_based_collaborative_filtering_rs import UserBasedCollaborativeFilteringRS, UserId, ItemId, Rating


def _get_ratings_set_data(set_path: str) -> Series:
    """
    Extracts the ratings given to items by users.

    :param set_path: The absolute path of the set file from which to get the data.
    :return: A series indexing the ratings by user_ids and item_ids.
    """
    ratings_df = pd.read_csv(set_path, delim_whitespace=True,
                             header=None, names=['user_id', 'item_id', 'rating', 'timestamp'],
                             index_col=['user_id', 'item_id'],
                             usecols=['user_id', 'item_id', 'rating'],
                             dtype={'user_id': UserId, 'item_id': ItemId, 'rating': Rating},
                             squeeze=True)
    return ratings_df


def _get_ratings_set_test_data(set_path: str) -> Series:
    """
    Extracts only the user/item pairs for whom the model will have to predict a rating.

    :param set_path: The absolute path of the set file from which to get only the user_ids and item_ids.
    :return: An empty series indexing the user_ids and item_ids for whom the model will have to predict a rating.
    """
    ratings_df = pd.read_csv(set_path, delim_whitespace=True,
                             header=None, names=['user_id', 'item_id', 'rating', 'timestamp'],
                             index_col=['user_id', 'item_id'],
                             usecols=['user_id', 'item_id'],
                             dtype={'user_id': UserId, 'item_id': ItemId},
                             squeeze=True)
    return ratings_df


def _evaluate_predictions_with_rmse(prediction: Series, groundtruth: Series) -> float:
    """
    Computes the Root Mean Square Error (RMSE) quality measure on the ratings predicted by the model for the set.

    :param prediction: The ratings predicted by the model for the set.
    :param groundtruth: The grountruth ratings for the set.
    :return: The Root Mean Square Error (RMSE) quality measure on the ratings predicted by the model for the set.
    """
    return ((prediction - groundtruth) ** 2).mean() ** .5


def test_user_based_prediction_model(sets_dir: str, knn: int, corr_min_periods: int, results: str):
    """
    Tests the user based collaborative filtering recommender system on each of the test sets, measuring the performance
    of the system on each of them and in average.

    :param sets_dir: The path of the directory of the sets files.
    :param knn: The number of nearest neighbors to take into account when making recommendations.
    :param corr_min_periods: The minimum number of items rated by both users to take into account the correlation
                             between them.
    :param results: The relative path of the output file compiling the Root Mean Square Error (RMSE) statistics of the
                    predictions.
    """
    train_sets_files = ['u1.base', 'u2.base', 'u3.base', 'u4.base', 'u5.base', ]
    test_sets_files = ['u1.test', 'u2.test', 'u3.test', 'u4.test', 'u5.test', ]

    rmse: Dict[str, float] = {}
    for train_set_file, test_set_file in zip(train_sets_files, test_sets_files):
        set_id = train_set_file.split(".")[0]
        train_set = _get_ratings_set_data(os.path.join(sets_dir, train_set_file))
        train_set.head()
        test_set = _get_ratings_set_data(os.path.join(sets_dir, test_set_file))
        test_set.head()

        model = UserBasedCollaborativeFilteringRS(knn, corr_min_periods)

        print(f'Training the model on the {train_set_file} set...')
        model.train(train_set)
        print(f'Finished training the model on the {train_set_file} set')

        print(f'Predicting ratings with the trained model on the {test_set_file} set...')
        prediction = model.predict(_get_ratings_set_test_data(os.path.join(sets_dir, test_set_file)))

        print(f'Evaluating the predictions of the model on the {test_set_file} set...')
        rmse[set_id] = _evaluate_predictions_with_rmse(prediction, test_set)

    with open(results, 'w') as f:
        f.writelines((f'Set: {set_id}  RMSE: {rmse}\n' for set_id, rmse in rmse.items()))
        f.write(f'Mean RMSE: {mean(rmse.values())}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Script that extracts users' movie ratings data from the MovieLens data sets and "
                                     "measures the accuracy of the predictions of a user based rating prediction model "
                                     "using the RSME metric.")
    parser.add_argument("--sets-dir", "-d", type=str, required=True,
                        help="The path of the directory of the sets files.")
    parser.add_argument("--neighbors", "-k", type=int, nargs='?', default=10,
                        help="The number of nearest neighbors to take into account when making recommendations.")
    parser.add_argument("--min-ratings", "-m", type=int, nargs='?', default=5,
                        help="The minimum number of items rated by both users to take into account the correlation "
                             "between them.")
    parser.add_argument("--results", "-r", type=str, nargs='?', default="results.txt",
                        help="The relative path of the output file compiling the Root Mean Square Error (RMSE) "
                             "statistics of the predictions.")
    args = parser.parse_args()

    test_user_based_prediction_model(args.sets_dir, args.neighbors, args.min_ratings, args.results)
