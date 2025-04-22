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
rf_model = RandomForestRegressor(max_features=0.5)
rf_model.fit(train_X, train_y)

util.eval_model(rf_model, train_X, train_y, test_X, test_y, "Random Forest")
print()

# Create forest model
rf = dforest.make_forest_sklearn(rf_model)

# Start timer
start_time = time.time()

# Run queries
n_trials = 200
samples = test_X[:n_trials]
delta = np.std(train_X, 0) / 20
eps = 0.8
hyperparams = [
    {"pso_N": 10_000, "pso_max_iters": 5, "merge_limit": 5},
    {"pso_N": 20_000, "pso_max_iters": 5, "merge_limit": 3, "offset_factor": 0.05}
]

run_robustness_evals(rf, samples, delta, eps, hyperparams)

# Free forest
rf.free()
