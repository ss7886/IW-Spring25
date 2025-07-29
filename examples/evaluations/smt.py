from argparse import ArgumentParser
import os
import random
import sys

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import examples.util as util
import dforest
from examples.evaluations.eval import run_smt_evals

def smt_eval(model, n_trials=10, stds=1/20, eps=0.8, gb=False,
                    output=None, timeout=500, np_seed=None, skl_seed=None):
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

    if output is not None:
        with open(output, "a") as file:
            file.write(f"Training shape: {train_X.shape}\n")
            file.write(f"Testing shape: {test_X.shape}\n\n")

    # Train scikit-learn model
    model.fit(train_X, train_y)

    util.eval_model(model, train_X, train_y, test_X, test_y, output=output)

    # Run queries
    samples = test_X[:n_trials]
    delta = stds * np.std(train_X, 0)

    run_smt_evals(model, samples, delta, eps, gb=gb, timeout=timeout,
                  output=output)

def rf_eval(output, timeout=300, n_trials=10, n_features=0.5,
            max_depth=None):
    if max_depth is None:
        model = RandomForestRegressor(max_features=n_features)
    else:
        model = RandomForestRegressor(max_features=n_features,
                                      max_depth=max_depth)

    smt_eval(model, timeout=timeout, n_trials=n_trials, output=output)

def bagging_eval(output, timeout=300, n_trials=10, max_depth=None):
    if max_depth is None:
        model = RandomForestRegressor()
    else:
        model = RandomForestRegressor(max_depth=max_depth)

    smt_eval(model, timeout=timeout, n_trials=n_trials, output=output)

def gb_eval(output, timeout=300, n_trials=10, max_depth=3):
    model = GradientBoostingRegressor(max_depth=max_depth)

    smt_eval(model, timeout=timeout, n_trials=n_trials, output=output, gb=True)

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
    parser.add_argument("-n", "--n_trials", default=10, type=int,
                        help="Number of trials to run")
    parser.add_argument("-t", "--timeout", default=300, type=int,
                        help="Z3 solver timeout (in seconds)")
    args = vars(parser.parse_args())
    return args

def main():
    args = handle_args()
    n = args["n_trials"]
    timeout = args["timeout"]
    if True not in args.values():
        print("No evals to run. Try running with flag --all or use flag "
              "--help to see full list of available evals.")
        return
    
    if not os.path.exists("./out/"):
        os.mkdir("./out")

    print("Running California Housing (SMT) evals")

    if args["rf"] or args["all"]:
        rf_eval("out/_smt_rf.out", timeout=timeout, n_trials=n)
        print("Finished Random Forest")
    
    if args["rf_15"] or args["all"]:
        rf_eval("out/_smt_rf_d15.out", max_depth=15, timeout=timeout,
                n_trials=n)
        print("Finished Random Forest (Max Depth = 15)")

    if args["bag"] or args["all"]:
        bagging_eval("out/_smt_bag.out", n_trials=n)
        print("Finished Bagging")
    
    if args["bag_15"] or args["all"]:
        rf_eval("out/_smt_bag_d15.out", max_depth=15, timeout=timeout,
                n_trials=n)
        print("Finished Bagging (Max Depth = 15)")

    if args["gb_8"] or args["all"]:
        gb_eval("out/_smt_gb_d8.out", timeout=timeout,
                max_depth=8, n_trials=n)
        print("Finished Gradient Boosting (Max Depth = 8)")

    if args["gb_5"] or args["all"]:
        gb_eval("out/_smt_gb_d5.out", timeout=timeout,
                max_depth=5, n_trials=n)
        print("Finished Gradient Boosting (Max Depth = 5)")

if __name__ == "__main__":
    main()
