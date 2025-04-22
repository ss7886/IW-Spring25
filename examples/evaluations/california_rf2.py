import time
import random

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor

import examples.util as util
import dforest
from examples.evaluations.eval import run_robustness_evals

np_seed = random.randint(0, 10_000)
skl_seed = 1234

print(f"numpy seed: {np_seed}")
print(f"scikit-learn seed: {skl_seed}")

np.random.seed(np_seed)

# Fetch Dataset
data_X, data_y = fetch_california_housing(return_X_y=True)

# Shuffle and split data
train_X, train_y, test_X, test_y = util.split(data_X, data_y, seed=skl_seed)

# Train scikit-learn model
rf_model2 = RandomForestRegressor(max_features=0.5, max_depth=15)
rf_model2.fit(train_X, train_y)

util.eval_model(rf_model2, train_X, train_y, test_X, test_y,
                "Random Forest (Max Depth = 15)")
print()

# Create forest model
rf2 = dforest.make_forest_sklearn(rf_model2)

# Start timer
start_time = time.time()

# Run queries
n_trials = 200
samples = test_X[:n_trials]
delta = np.std(train_X, 0) / 20
eps = 0.8
hyperparams = [
    {"pso_N": 10_000, "pso_max_iters": 5, "merge_limit": 5},
    {"pso_N": 40_000, "pso_max_iters": 5, "merge_limit": 2}
]

run_robustness_evals(rf2, samples, delta, eps, hyperparams)

# Free forest
rf2.free()
