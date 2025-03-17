from dtree import *

import numpy as np
import pytest
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

def test_eval_matrix():
    leaf_a = make_leaf(2, 0.0)
    leaf_b = make_leaf(2, 1.0)
    tree = make_split(2, 0, 0.5, leaf_a, leaf_b)

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype='float64')
    y = tree.eval(X)
    assert np.array_equal(y, np.array([0, 0, 1, 1], dtype='float64'))

    tree.free()

def test_copy():    
    leaf_a = make_leaf(2, 0.0)
    leaf_b = make_leaf(2, 1.0)
    tree = make_split(2, 0, 0.5, leaf_a, leaf_b)
    copy = tree.copy()

    assert tree.eval([0, 0]) == copy.eval([0, 0])
    assert tree.eval([1, 1]) == copy.eval([1, 1])
    assert tree.eval([0.5, 0.5]) == copy.eval([0.5, 0.5])

    tree.free()

    assert copy.eval([0, 0]) == 0.0
    assert copy.eval([1, 1]) == 1.0
    assert copy.eval([0.5, 0.5]) == 0.0

    copy.free()

def test_prune_left():
    splita = make_split(2, 1, 0.25, make_leaf(2, 1), make_leaf(2, 2))
    splitb = make_split(2, 1, 0.75, make_leaf(2, 3), make_leaf(2, 4))
    tree = make_split(2, 0, 0.5, splita, splitb)

    prune1 = tree.copy().prune_left(0, 0.6)
    prune2 = tree.copy().prune_left(0, 0.4)
    prune3 = tree.copy().prune_left(1, 0.5)

    assert prune1.depth == 2
    assert prune1.size == 4
    assert prune2.depth == 1
    assert prune2.size == 2
    assert prune3.depth == 2
    assert prune3.size == 3

    tree.free()
    prune1.free()
    prune2.free()
    prune3.free()

def test_prune_right():
    splita = make_split(2, 1, 0.25, make_leaf(2, 1), make_leaf(2, 2))
    splitb = make_split(2, 1, 0.75, make_leaf(2, 3), make_leaf(2, 4))
    tree = make_split(2, 0, 0.5, splita, splitb)

    prune1 = tree.copy().prune_right(0, 0.6)
    prune2 = tree.copy().prune_right(0, 0.4)
    prune3 = tree.copy().prune_right(1, 0.5)

    assert prune1.depth == 1
    assert prune1.size == 2
    assert prune2.depth == 2
    assert prune2.size == 4
    assert prune3.depth == 2
    assert prune3.size == 3

    tree.free()
    prune1.free()
    prune2.free()
    prune3.free()

def test_prune():
    splita = make_split(2, 1, 0.25, make_leaf(2, 1), make_leaf(2, 2))
    splitb = make_split(2, 1, 0.75, make_leaf(2, 3), make_leaf(2, 4))
    tree = make_split(2, 0, 0.5, splita, splitb)

    tree.prune_left(1, 0.6)
    tree.prune_right(1, 0.4)

    assert tree.depth == 1
    assert tree.size == 2
    assert tree.min == 2.0
    assert tree.max == 3.0

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
    Check that a tree merged with itself has the same shape.
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

def test_feature_importance():
    splitRL = make_split(3, 0, 0, make_leaf(3, 0), make_leaf(3, 0))
    splitRR = make_split(3, 2, 0, make_leaf(3, 0), make_leaf(3, 0))
    splitL = make_split(3, 2, 0, make_leaf(3, 0), make_leaf(3, 0))
    splitR = make_split(3, 0, 0, splitRL, splitRR)
    tree = make_split(3, 1, 0, splitL, splitR)

    importances = tree.feature_importance()
    assert np.array_equal(importances, np.array([0.3125, 0.375, 0.3125]))

    tree.free()

def test_free():
    leaf_a = make_leaf(2, 0.0)
    leaf_b = make_leaf(2, 1.0)
    tree = make_split(2, 0, 0.5, leaf_a, leaf_b)
    tree.free()

    with pytest.raises(treeCFFIError):
        tree.eval([0, 0]) == 0.0
    with pytest.raises(treeCFFIError):
        tree.free()
    with pytest.raises(treeCFFIError):
        tree.copy()
    with pytest.raises(treeCFFIError):
        tree.prune_left(0, 0)
    with pytest.raises(treeCFFIError):
        tree.prune_right(0, 0)
    with pytest.raises(treeCFFIError):
        merge_trees(tree, tree)
    with pytest.raises(treeCFFIError):
        tree.feature_importance()

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
