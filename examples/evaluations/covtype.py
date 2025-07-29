from argparse import ArgumentParser
import os
import random
import sys

import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import examples.util as util
import dforest_class
from examples.evaluations.eval import run_multiclass_evals

def covtype_eval(model, hyperparams, n_trials=200, n_samples=50_000, stds=1/20,
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

    # Load dataset
    data_X, data_y = fetch_covtype(return_X_y=True, shuffle=True)
    data_X, data_y = data_X[:n_samples], data_y[:n_samples]

    # Shuffle and split data
    train_X, train_y, test_X, test_y = util.split(data_X, data_y, seed=skl_seed)

    if output is not None:
        with open(output, "a") as file:
            file.write(f"Training shape: {train_X.shape}\n")
            file.write(f"Testing shape: {test_X.shape}\n\n")

    # Train scikit-learn model
    model.fit(train_X, train_y)

    util.eval_multiclass_model(model, train_X, train_y, test_X, test_y,
                               output=output)

    # Create forest model
    forest = dforest_class.make_forest_classifier_sklearn(model, gb=gb)

    if output is not None:
        with open(output, "a") as file:
            file.write(f"# classes: {forest.n_class}\n")
            file.write(f"# trees: {forest.n_trees}\n")
            file.write("Avg size:\n")
            for class_id in range(forest.n_class):
                file.write(f"\tClass {forest.classes[class_id]}: {forest.avg_size(class_id)}\n")
            file.write("\n")

    # Run queries
    samples = test_X[:n_trials]
    delta = np.std(train_X, axis=0) * stds
    clip_min = np.min(data_X, axis=0)
    clip_max = np.max(data_X, axis=0)

    run_multiclass_evals(forest, samples, delta, hyperparams, clip_min, clip_max,
                         output=output)

    # Free forest
    forest.free()

default_params = [
    {"pso_N": 20_000, "pso_max_iters": 5, "merge_limit": 2}
]

strong_params = [
    {"pso_N": 40_000, "pso_max_iters": 5, "merge_limit": 2}
]

def rf_eval(output, hyperparams=None, n_trials=200, n_features="sqrt",
            max_depth=None):
    if max_depth is None:
        model = RandomForestClassifier(max_features=n_features)
    else:
        model = RandomForestClassifier(max_features=n_features,
                                       max_depth=max_depth)

    if hyperparams is None:
        hyperparams = default_params

    covtype_eval(model, hyperparams, n_trials=n_trials, output=output)

def bagging_eval(output, hyperparams=None, n_trials=200, max_depth=None):
    if max_depth is None:
        model = RandomForestClassifier(max_features=None)
    else:
        model = RandomForestClassifier(max_features=None,
                                       max_depth=max_depth)

    if hyperparams is None:
        hyperparams = default_params

    covtype_eval(model, hyperparams, n_trials=n_trials, output=output)

def gb_eval(output, hyperparams=None, n_trials=200, max_depth=3):
    model = GradientBoostingClassifier(max_depth=max_depth)

    if hyperparams is None:
        hyperparams = default_params

    covtype_eval(model, hyperparams, n_trials=n_trials, output=output, gb=True)

def handle_args():
    """
    Handle and return arguments using ArgumentParser.
    """
    parser = ArgumentParser(prog=sys.argv[0],
                            description="Build C code with CFFI.",
                            allow_abbrev=False)
    parser.add_argument("--rf", action="store_true",
                        help="Random Forest")
    parser.add_argument("--rf_20", action="store_true",
                        help="Random Forest (Max Depth = 20)")
    parser.add_argument("--bag", action="store_true",
                        help="Bagging")
    parser.add_argument("--bag_18", action="store_true",
                        help="Bagging (Max Depth = 18)")
    parser.add_argument("--gb_12", "--gb", action="store_true",
                        help="Gradient Boosting (Max Depth = 12)")
    parser.add_argument("--gb_8", action="store_true",
                        help="Gradient Boosting (Max Depth = 8)")
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

    print("Running Forest Covtype evals")

    if args["rf"] or args["all"]:
        rf_eval("out/_covtype_rf.out", n_trials=n)
        print("Finished Random Forest")
    
    if args["rf_20"] or args["all"]:
        rf_eval("out/_covtype_rf_d20.out", max_depth=20, n_trials=n)
        print("Finished Random Forest (Max Depth = 20)")

    if args["bag"] or args["all"]:
        bagging_eval("out/_covtype_bag.out", n_trials=n)
        print("Finished Bagging")
    
    if args["bag_18"] or args["all"]:
        rf_eval("out/_covtype_bag_d18.out", max_depth=18, n_trials=n)
        print("Finished Bagging (Max Depth = 18)")

    if args["gb_12"] or args["all"]:
        gb_eval("out/_covtype_gb_d12.out", hyperparams=strong_params,
                max_depth=12, n_trials=n)
        print("Finished Gradient Boosting (Max Depth = 12)")

    if args["gb_8"] or args["all"]:
        gb_eval("out/_covtype_gb_d8.out", hyperparams=strong_params,
                max_depth=8, n_trials=n)
        print("Finished Gradient Boosting (Max Depth = 8)")

if __name__ == "__main__":
    main()
