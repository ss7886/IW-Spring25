import gc

from dtree import *
from dforest import *

import numpy as np

def test_make_forest():
    trees = []
    trees.append(make_split(2, 0, 0, make_leaf(2, 0), make_leaf(2, 1)))
    trees.append(make_split(2, 1, 0, make_leaf(2, 0), make_leaf(2, 1)))
    forest = Forest(trees, 1)

    assert forest.eval([-1, -1]) == 0
    assert forest.eval([-1, 1]) == 1
    assert forest.eval([1, -1]) == 1
    assert forest.eval([1, 1]) == 2

    forest.free()

def test_eval_matrix():
    trees = []
    trees.append(make_split(2, 0, 0, make_leaf(2, 0), make_leaf(2, 1)))
    trees.append(make_split(2, 1, 0, make_leaf(2, 0), make_leaf(2, 1)))
    forest = Forest(trees, 1)

    X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype='float64')
    y = forest.eval(X)
    assert np.array_equal(y, np.array([0, 1, 1, 2], dtype='float64'))

    forest.free()
