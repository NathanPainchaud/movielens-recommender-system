#!/usr/bin/fish

# Activate the Python environment, in this example a conda environment
conda activate data_mining

# Initialize results directory
rm -rf results
for k in (seq 1 10)
    mkdir -p results/k$k
end

# Search for the best hyperparameter combination between
# - k: The number of nearest neighbors to take into account when making recommendations;
# - m: The minimum number of items rated by both users to take into account the correlation between them.
for k in (seq 1 10)
    for m in (seq 1 10)
        python collaborative_filtering_rs.py --sets-dir "/data/data_mining/MovieLens/100K/ml-100k" \
        --neighbors $k \
        --min-ratings $m \
        --results "results/k$k/m$m.txt"
    end
end
