import argparse
import os
import sys
from shutil import copyfile


def find_best_hyperparameters(results_dir: str):
    """
    Walks through the results directory hierarchy to evaluate the results of each recommender system's performance, and
    copies the results of the best system to the root of the hierarchy.

    NOTE: To function properly, this script assumes that the structure of the directory follows the same principle of
    folders for number of neighbors and file for number of common ratings as used in the `run.sh` example.

    :param results_dir: The path of the root directory of the results to search.
    """

    def _get_result_mean_rmse(result_file: str) -> float:
        """
        Extracts the  mean RMSE of the recommender system over all the test sets from the detailed performance of the
        recommender system.

        :param result_file: The path of the file detailing the performance of the recommender system on each of the test
                            sets and on average.
        :return: The mean RMSE of the recommender system over all the test sets.
        """
        with open(result_file, 'r') as f:
            mean_rmse = float(f.read().splitlines()[-1].split()[2])
        return mean_rmse

    best_rmse = sys.maxsize
    for root, dir, files in os.walk(results_dir):
        for file in files:
            k = int(root.split(os.sep)[-1][1:])
            m = int(file.split('.')[0][1:])
            rmse = _get_result_mean_rmse(os.path.join(root, file))
            if rmse < best_rmse:
                best_rmse = rmse
                best_system = k, m, os.path.join(root, file)

    best_k, best_m, best_result = best_system
    copyfile(best_result, os.path.join(results_dir, f'k{best_k}_m{best_m}.txt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Script that walks through the results directory hierarchy to evaluate the "
                                     "results of each recommender system's performance, and copies the results of the "
                                     "best system to the root of the hierarchy.")
    parser.add_argument("--results-dir", "-r", type=str, nargs='?', default="results",
                        help="The path of the root directory of the results to search.")
    args = parser.parse_args()

    find_best_hyperparameters(args.results_dir)
