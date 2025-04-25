from argparse import ArgumentParser
import os
import random
import sys

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import examples.util as util
import dforest
from examples.evaluations.eval import run_robustness_evals

def california_eval(model, hyperparams, n_trials=200, stds=1/20, eps=0.8,
                    output=None, np_seed=None, skl_seed=None, gb=False):
    if np_seed is None:
        np_seed = random.randint(0, 10_000)
    if skl_seed is None:
        skl_seed = 1234

    if output is not None:
        with open(output, "w") as file:
            file.write(f"numpy seed: {np_seed}\n")
            file.write(f"scikit-learn seed: {skl_seed}\n\n")

    np.random.seed(np_seed)

    # Fetch Dataset
    data_X, data_y = fetch_california_housing(return_X_y=True)

    # Shuffle and split data
    train_X, train_y, test_X, test_y = util.split(data_X, data_y, seed=skl_seed)

    # Train scikit-learn model
    model.fit(train_X, train_y)

    util.eval_model(model, train_X, train_y, test_X, test_y,
                    "Random Forest", output)

    # Create forest model
    rf = dforest.make_forest_sklearn(model, gb=gb)

    # Run queries
    samples = test_X[:n_trials]
    delta = stds * np.std(train_X, 0)

    run_robustness_evals(rf, samples, delta, eps, hyperparams, output=output)

    # Free forest
    rf.free()

default_params = [
    {"pso_N": 10_000, "pso_max_iters": 5, "merge_limit": 5},
    {"pso_N": 20_000, "pso_max_iters": 5, "merge_limit": 3}
]

strong_params = [
    {"pso_N": 10_000, "pso_max_iters": 5, "merge_limit": 5},
    {"pso_N": 40_000, "pso_max_iters": 5, "merge_limit": 2}
]

def rf_eval(output, hyperparams=None, n_trials=200, n_features=0.5,
            max_depth=None):
    if max_depth is None:
        model = RandomForestRegressor(max_features=n_features)
    else:
        model = RandomForestRegressor(max_features=n_features,
                                      max_depth=max_depth)

    if hyperparams is None:
        hyperparams = default_params

    california_eval(model, hyperparams, n_trials=n_trials, output=output)

def bagging_eval(output, hyperparams=None, n_trials=200, max_depth=None):
    if max_depth is None:
        model = RandomForestRegressor()
    else:
        model = RandomForestRegressor(max_depth=max_depth)

    if hyperparams is None:
        hyperparams = default_params

    california_eval(model, hyperparams, n_trials=n_trials, output=output)

def gb_eval(output, hyperparams=None, n_trials=200, max_depth=3):
    model = GradientBoostingRegressor(max_depth=max_depth)

    if hyperparams is None:
        hyperparams = default_params

    california_eval(model, hyperparams, n_trials=n_trials, output=output, gb=True)

def handle_args():
    """
    Handle and return arguments using ArgumentParser.
    """
    parser = ArgumentParser(prog=sys.argv[0],
                            description="Build C code with CFFI.",
                            allow_abbrev=False)
    parser.add_argument("--rf", action="store_true",
                        help="Random Forest")
    parser.add_argument("--rf_15", action="store_true",
                        help="Random Forest (Max Depth = 15)")
    parser.add_argument("--bag", action="store_true",
                        help="Bagging")
    parser.add_argument("--bag_15", action="store_true",
                        help="Bagging (Max Depth = 15)")
    parser.add_argument("--gb_8", "--gb", action="store_true",
                        help="Gradient Boosting (Max Depth = 8)")
    parser.add_argument("--gb_5", action="store_true",
                        help="Gradient Boosting (Max Depth = 5)")
    parser.add_argument("--all", action="store_true",
                        help="Run all tests")
    parser.add_argument("-n", "--n_trials", default=200, type=int,
                        help="Number of trials to run")
    args = vars(parser.parse_args())
    return args

def main():
    args = handle_args()
    n = args["n_trials"]
    if True not in args.values():
        print("No evals to run. Try running with flag --all or use flag "
              "--help to see full list of available evals.")
        return
    
    if not os.path.exists("./out/"):
        os.mkdir("./out")

    print("Running California Housing evals")

    if args["rf"] or args["all"]:
        rf_eval("out/_california_rf.out", n_trials=n)
        print("Finished Random Forest")
    
    if args["rf_15"] or args["all"]:
        rf_eval("out/_california_rf_d15.out", max_depth=15, n_trials=n)
        print("Finished Random Forest (Max Depth = 15)")

    if args["bag"] or args["all"]:
        bagging_eval("out/_california_bag.out", n_trials=n)
        print("Finished Bagging")
    
    if args["bag_15"] or args["all"]:
        rf_eval("out/_california_rf_d15.out", max_depth=15, n_trials=n)
        print("Finished Bagging (Max Depth = 15)")

    if args["gb_8"] or args["all"]:
        gb_eval("out/_california_gb_d8.out", hyperparams=strong_params,
                max_depth=8, n_trials=n)
        print("Finished Gradient Boosting (Max Depth = 8)")

    if args["gb_5"] or args["all"]:
        gb_eval("out/_california_gb_d5.out", hyperparams=strong_params,
                max_depth=5, n_trials=n)
        print("Finished Gradient Boosting (Max Depth = 5)")

if __name__ == "__main__":
    main()
