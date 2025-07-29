from argparse import ArgumentParser
import os
import random
import sys

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import examples.util as util
import dforest_class
from examples.evaluations.eval import run_multiclass_evals

def mnist_eval(model, hyperparams, n_trials=200, output=None, np_seed=None,
                 skl_seed=None, gb=False):
    if np_seed is None:
        np_seed = random.randint(0, 10_000)
    if skl_seed is None:
        skl_seed = 1234

    if output is not None:
        with open(output, "w") as file:
            file.write(f"numpy seed: {np_seed}\n")
            file.write(f"scikit-learn seed: {skl_seed}\n\n")

    np.random.seed(np_seed)

    # Load full MNIST from OpenML
    df_X, df_y = fetch_openml('mnist_784', return_X_y=True, version=1)

    # Resize images with max-pooling
    resized = []
    for i in range(df_X.shape[0]):
        im = util.image_resize(df_X.iloc[i].values.reshape((28, 28)))
        resized.append(im.flatten())
    data_X = np.array(resized)
    data_y = np.array(df_y)
    brightness_scale = 64
    data_X = data_X // (256 / brightness_scale)

    # Shuffle and split data
    train_X, train_y, test_X, test_y = util.split(data_X, data_y, seed=skl_seed)

    if output is not None:
        with open(output, "a") as file:
            file.write(f"Training shape: {train_X.shape}\n")
            file.write(f"Testing shape: {test_X.shape}\n\n")

    # Train scikit-learn model
    model.fit(train_X, train_y)

    print("Finished Training")

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
    delta = np.ones(14 * 14)
    clip_min = 0
    clip_max = brightness_scale

    run_multiclass_evals(forest, samples, delta, hyperparams, clip_min, clip_max,
                         output=output)

    # Free forest
    forest.free()

default_params = [
    {"pso_N": 10_000, "pso_max_iters": 5, "merge_limit": 12}
]

strong_params = [
    {"pso_N": 20_000, "pso_max_iters": 5, "merge_limit": 8}
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

    mnist_eval(model, hyperparams, n_trials=n_trials, output=output)

def bagging_eval(output, hyperparams=None, n_trials=200, max_depth=None):
    if max_depth is None:
        model = RandomForestClassifier(max_features=None)
    else:
        model = RandomForestClassifier(max_features=None, max_depth=max_depth)

    if hyperparams is None:
        hyperparams = default_params

    mnist_eval(model, hyperparams, n_trials=n_trials, output=output)

def gb_eval(output, hyperparams=None, n_trials=200, max_depth=3):
    model = GradientBoostingClassifier(max_depth=max_depth)

    if hyperparams is None:
        hyperparams = default_params

    mnist_eval(model, hyperparams, n_trials=n_trials, output=output, gb=True)

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
    parser.add_argument("--bag_12", action="store_true",
                        help="Bagging (Max Depth = 12)")
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

    print("Running MNIST Digits evals")

    if args["rf"] or args["all"]:
        rf_eval("out/_mnist_rf.out", n_trials=n)
        print("Finished Random Forest")
    
    if args["rf_15"] or args["all"]:
        rf_eval("out/_mnist_rf_d15.out", max_depth=15, n_trials=n)
        print("Finished Random Forest (Max Depth = 15)")

    if args["bag"] or args["all"]:
        bagging_eval("out/_mnist_bag.out", n_trials=n)
        print("Finished Bagging")
    
    if args["bag_12"] or args["all"]:
        rf_eval("out/_mnist_bag_d12.out", max_depth=12, n_trials=n)
        print("Finished Bagging (Max Depth = 12)")

    if args["gb_8"] or args["all"]:
        gb_eval("out/_mnist_gb_d8.out", hyperparams=strong_params,
                max_depth=8, n_trials=n)
        print("Finished Gradient Boosting (Max Depth = 8)")

    if args["gb_5"] or args["all"]:
        gb_eval("out/_mnist_gb_d5.out", hyperparams=strong_params,
                max_depth=5, n_trials=n)
        print("Finished Gradient Boosting (Max Depth = 5)")

if __name__ == "__main__":
    main()
