#!/usr/bin/fish

# Activate the Python environment, in this example a conda environment
conda activate data_mining

# Initialize results directory
rm -rf results
for k in (seq 1 30)
    mkdir -p results/k$k
end

# Test various hyperparameters' combinations between
# - k: The number of nearest neighbors to take into account when making recommendations;
# - m: The minimum number of items rated by both users to take into account the correlation between them.
for k in (seq 1 30)
    for m in (seq 1 30)
        python movielens_recommender_system.py --sets-dir "/data/data_mining/MovieLens/100K/ml-100k" \
        --neighbors $k \
        --min-ratings $m \
        --results "results/k$k/m$m.txt"
    end
end

# Search through the results for the hyperparameters' combination with the smallest RMSE on average
python cross_validation.py --results-dir "results"
