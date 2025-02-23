from tree import make_leaf, make_split, make_tree_sklearn, merge_trees

import numpy as np
# import pytest
from sklearn.tree import DecisionTreeRegressor

def test_make_tree():
    leaf_a = make_leaf(2, 0.0)
    leaf_b = make_leaf(2, 1.0)
    tree = make_split(2, 0, 0.5, leaf_a, leaf_b)

    assert tree.eval([0, 0]) == 0.0
    assert tree.eval([0, 1]) == 0.0
    assert tree.eval([1, 0]) == 1.0
    assert tree.eval([1, 1]) == 1.0
    assert tree.eval([0.5, 0.5]) == 0.0

    assert tree.eval(np.array([0, 0])) == 0.0
    assert tree.eval(np.array([1, 1])) == 1.0

    tree.free()

def test_merge():
    split_1a = make_split(2, 1, 0.25, make_leaf(2, 1), make_leaf(2, 2))
    split_1b = make_split(2, 1, 0.75, make_leaf(2, 3), make_leaf(2, 4))
    tree1 = make_split(2, 0, 0.5, split_1a, split_1b)

    split_2a = make_split(2, 0, 0.25, make_leaf(2, 1), make_leaf(2, 3))
    split_2b = make_split(2, 0, 0.75, make_leaf(2, 2), make_leaf(2, 4))
    tree2 = make_split(2, 1, 0.5, split_2a, split_2b)

    merge = merge_trees(tree1, tree2)

    assert merge.depth == 4
    assert merge.size == 10
    assert merge.min == 2.0
    assert merge.max == 8.0

    assert merge.eval([0.1, 0.1]) == 2.0
    assert merge.eval([0.3, 0.1]) == 4.0
    assert merge.eval([0.1, 0.3]) == 3.0
    assert merge.eval([0.3, 0.3]) == 5.0
    assert merge.eval([0.2, 0.7]) == 4.0
    assert merge.eval([0.7, 0.2]) == 6.0
    assert merge.eval([0.6, 0.6]) == 5.0
    assert merge.eval([0.6, 0.8]) == 6.0
    assert merge.eval([0.8, 0.6]) == 7.0
    assert merge.eval([0.8, 0.8]) == 8.0

    tree1.free()
    tree2.free()
    merge.free()

def test_merge_self():
    """
    Check that 
    """
    leaf_a = make_leaf(2, 0.0)
    leaf_b = make_leaf(2, 1.0)
    tree = make_split(2, 0, 0.5, leaf_a, leaf_b)

    merge = merge_trees(tree, tree)

    for i in range(10):
        old_merge = merge
        merge = merge_trees(merge, tree)
        old_merge.free()
    
    assert merge.size == 2
    assert merge.depth == 1
    assert merge.min == 0.0
    assert merge.max == 12.0

    assert merge.eval([0, 0]) == 0.0
    assert merge.eval([1, 1]) == 12.0
    assert merge.eval([0.5, 0.5]) == 0.0

    tree.free()
    merge.free()

def test_auto_dataset():
    missing_data = lambda x : 100. if x == '?' else float(x)
    data = np.loadtxt("datasets/Auto.data", converters=missing_data, 
                      skiprows=1, usecols=[0, 1, 2, 3, 4, 5, 6, 7])

    np.random.seed(12345)
    np.random.shuffle(data)

    auto_X = data[:, 1:]
    auto_y = data[:, 0]

    model = DecisionTreeRegressor()
    model.fit(auto_X, auto_y)
    tree = make_tree_sklearn(model)

    assert tree.depth == model.get_depth()
    assert tree.size == model.get_n_leaves()
    assert tree.min == np.min(model.tree_.value)
    assert tree.max == np.max(model.tree_.value)

    model_preds = model.predict(auto_X)
    tree_preds = []
    for x in auto_X:
        tree_preds.append(tree.eval(x))
    for i in range(len(model_preds)):
        assert model_preds[i] == tree_preds[i]
    
    tree.free()
