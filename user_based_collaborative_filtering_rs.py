import sys
from typing import List, Tuple

import numpy as np
import pandas as pd

from pandas import Series
from tqdm import tqdm


UserId = np.uint16
ItemId = np.uint16
Rating = np.float


class UserBasedCollaborativeFilteringRS:
    """
    Class that implements a user based collaborative filtering recommender system.
    """

    correlations: Series
    stds: Series
    means: Series
    ratings: Series

    def __init__(self, knn: int=5, min_ratings: int=1):
        """
        Initializes the hyperparameters of the recommender system.

        :param knn: The number of k-nearest neighbors to base the recommendations on.
        :param min_ratings: The minimum number of items rated by both users to take into account the correlation
                            between them.
        """
        self.knn = knn
        self.min_ratings = min_ratings

    def train(self, users_ratings: Series):
        """
        Trains the model by:
        - normalizing the ratings it receives;
        - precomputing the correlations between every user

        :param users_ratings: A series indexing the ratings by user_ids and item_ids.
        """
        self.ratings = users_ratings
        self.ratings, self.means, self.stds = self._normalize_ratings(self.ratings)
        self.correlations = self._compute_correlations(self.ratings, self.min_ratings)

    def predict(self, users_ratings_to_predict: Series) -> Series:
        """
        For every user/item pair received in parameter, predicts the rating the user would give the item.

        :param users_ratings_to_predict: An empty series indexing the user_ids and item_ids for whom to predict a
                                         rating.
        :return: A series indexing the ratings predicted by the model by user_ids and item_ids.
        """

        def _select_neighbors(user_id: UserId, item_id: ItemId) -> Series:
            """
            Finds users most similar to the current user that have also rated the item for which to predict the current
            user's rating.

            :param user_id: The user for whom to find similar users.
            :param item_id: The item that similar users have to have rated.
            :return: A series indexing the correlation between the user and its k nearest neighbors by neighbor_id.
            """
            try:
                user_correlations = self.correlations.loc[user_id]
                neighbors = user_correlations.loc[(user_correlations.index.isin(self.ratings[:, item_id].index))]\
                    .head(self.knn)
            except KeyError:    # If no users similar to the current user have rated the item, return no neighbors
                neighbors = pd.Series()
            return neighbors

        def _predict_user_rating(user_id: UserId, item_id: ItemId) -> Rating:
            """
            Predicts the rating a user would give to an item.

            :param user_id: The user for whom to make the prediction.
            :param item_id: The item for which to predict the user's rating.
            :return: The rating predicted to be given to the item by the user.
            """
            mean: Rating = self.means[user_id]
            std: float = self.stds[user_id]

            neighbors = _select_neighbors(user_id, item_id)
            if not neighbors.empty:
                rating = sum(correlation * self.ratings.loc[neighbor_id, item_id]
                             for neighbor_id, correlation in neighbors.items()) \
                         / sum(abs(correlation) for _, correlation in neighbors.items())
                predicted_rating = (rating * std) + mean

            else:   # If no users similar to the current user have rated the item, default to mean user rating
                predicted_rating = mean

            return round(predicted_rating)

        prediction = users_ratings_to_predict
        tqdm.pandas(desc='Predicting ratings for every user/item pair in the set',
                    unit=' ratings',
                    file=sys.stdout)
        prediction = Series(prediction.reset_index().progress_apply(lambda row: _predict_user_rating(*row),
                                                                    axis=1).values,
                            index=prediction.index)
        return prediction

    @staticmethod
    def _normalize_ratings(ratings: Series) -> Tuple[Series, Series, Series]:
        """
        Normalizes the ratings by mean centering them and expressing them relative to the standard deviation of the
        user's ratings.

        :param ratings: A series indexing the ratings by user_ids and item_ids.
        :return: A tuple of three series:
                 - A series indexing the normalized ratings by user_ids and item_ids;
                 - A series indexing the mean ratings given by the users by user_id;
                 - A series indexing the standard deviation from the user's mean rating by user_ids.
        """

        print("Normalizing the ratings...")

        # Mean centering
        means = ratings.groupby(level='user_id').mean()
        ratings = ratings - means

        # Division by std
        stds = ratings.groupby(level='user_id').std()
        ratings = ratings / stds

        return ratings, means, stds

    @staticmethod
    def _compute_correlations(ratings: Series, min_ratings: int) -> Series:
        """
        Computes the correlations between every user, based on the items they have both rated.

        :param ratings: A series indexing the ratings by user_ids and item_ids
                        (it is recommended that the ratings be normalized at this step).
        :param min_ratings: The minimum number of items rated by both users to take into account the correlation
                            between them.
        :return: A series indexing the correlation between a user and other users by user_ids and (neighbors') user_ids.
        """

        ratings_matrix = ratings.unstack(level='user_id')
        correlation_matrix = ratings_matrix.corr(min_periods=min_ratings)

        def _compute_user_correlations(user_id: UserId):
            user_correlation = correlation_matrix[user_id].dropna().sort_values(ascending=False)
            user_correlation = pd.concat([user_correlation], keys=[user_id])
            user_correlation.index.names = ['user_id', 'neighbor_id']
            return user_correlation

        users_neighbors: List[Series] = []
        pbar = tqdm(ratings.index.get_level_values('user_id').unique(),
                    desc='Computing correlation between every user',
                    unit=' users',
                    file=sys.stdout)
        for user_id in pbar:
            users_neighbors.append(_compute_user_correlations(user_id))

        return pd.concat(users_neighbors)
