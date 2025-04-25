from typing import Iterable, Optional, Tuple, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from sklearn.tree import DecisionTreeRegressor
from sklearn import ensemble
import z3

if TYPE_CHECKING:
    Vector = Iterable[float]
else:
    Vector = object

def tree_constraint(reg: DecisionTreeRegressor, min_bound: Vector = None,
                    max_bound: Vector = None, factor: float = 1., 
                    offset: float = 0., treeId: str = "Tree",
                    X: Iterable[z3.Real] = None):
    dim = reg.n_features_in_
    children_left = reg.tree_.children_left
    children_right = reg.tree_.children_right
    feature = reg.tree_.feature
    threshold = reg.tree_.threshold
    values = reg.tree_.value

    if min_bound is None:
        min_bound = [float("-inf") for _ in range(dim)]
    else:
        assert len(min_bound) == dim
    if max_bound is None:
        max_bound = [float("inf") for _ in range(dim)]
    else:
        assert len(max_bound) == dim
    
    path_cons = []
    cons = []

    if X is None:
        X = [z3.Real(f"x{i}") for i in range(dim)]
    o = z3.Real(treeId)

    def process_node(id: int) -> None:
        if children_left[id] == children_right[id]:
            val = values[id][0][0]
            path_cons.append(o == (val * factor + offset))
            cons.append(z3.And(*path_cons))
            path_cons.pop()
            return
        
        if threshold[id] >= min_bound[feature[id]]:
            path_cons.append(X[feature[id]] <= threshold[id])
            process_node(children_left[id])
            path_cons.pop()

        if threshold[id] <= max_bound[feature[id]]:
            path_cons.append(X[feature[id]] > threshold[id])
            process_node(children_right[id])
            path_cons.pop()
        return
    
    process_node(0)

    return z3.Or(*cons), o

def forest_constraint(forest: ensemble, X: Iterable[z3.Real] = None, **kwargs):
    dim = forest.n_features_in_
    cons = []
    outputs = []

    if X is None:
        X = [z3.Real(f"x{i}") for i in range(dim)]
    
    for i, tree in enumerate(forest.estimators_):
        if isinstance(tree, np.ndarray):
            tree = tree[0]
        con, output = tree_constraint(tree, X=X, treeId=f"o{i}", **kwargs)
        cons.append(con)
        outputs.append(output)
    return z3.And(cons), outputs

def max_query(forest: ensemble, min_bound: Vector, max_bound: Vector,
              threshold: float, gb: bool = False) -> Optional[bool]:
    dim = forest.n_features_in_
    n_trees = forest.n_estimators

    X = [z3.Real(f"x{i}") for i in range(dim)]

    min_con = z3.And(*[X[i] >= min_bound[i] for i in range(dim)])
    max_con = z3.And(*[X[i] <= max_bound[i] for i in range(dim)])

    if gb:
        factor = forest.learning_rate
        offset = forest.init_.constant_[0, 0]
    else:
        factor = 1 / n_trees
        offset = 0
    forest_con, outputs = forest_constraint(forest, min_bound=min_bound,
                                            max_bound=max_bound, factor=factor,
                                            offset=offset, X=X)
    s = z3.Solver()
    s.add(forest_con, min_con, max_con)
    s.add(z3.Sum(*outputs) > threshold)
    res = s.check()
    return True if res == z3.unsat else False if res == z3.sat else None

def min_query(forest: ensemble, min_bound: Vector, max_bound: Vector,
              threshold: float, gb: bool = False) -> Optional[bool]:
    dim = forest.n_features_in_
    n_trees = forest.n_estimators

    X = [z3.Real(f"x{i}") for i in range(dim)]

    min_con = z3.And(*[X[i] >= min_bound[i] for i in range(dim)])
    max_con = z3.And(*[X[i] <= max_bound[i] for i in range(dim)])

    if gb:
        factor = forest.learning_rate
        offset = forest.init_.constant_[0, 0]
    else:
        factor = 1 / n_trees
        offset = 0
    forest_con, outputs = forest_constraint(forest, min_bound=min_bound,
                                            max_bound=max_bound, factor=factor,
                                            offset=offset, X=X)
    s = z3.Solver()
    s.add(forest_con, min_con, max_con)
    s.add(z3.Sum(*outputs) < threshold)
    res = s.check()
    return True if res == z3.unsat else False if res == z3.sat else None

def query(forest: ensemble, min_bound: Vector, max_bound: Vector,
          min_threshold: float, max_threshold: float,
          gb: bool = False) -> Optional[bool]:
    dim = forest.n_features_in_
    n_trees = forest.n_estimators

    X = [z3.Real(f"x{i}") for i in range(dim)]

    min_con = z3.And(*[X[i] >= min_bound[i] for i in range(dim)])
    max_con = z3.And(*[X[i] <= max_bound[i] for i in range(dim)])

    if gb:
        factor = forest.learning_rate
        offset = forest.init_.constant_[0, 0]
    else:
        factor = 1 / n_trees
        offset = 0
    forest_con, outputs = forest_constraint(forest, min_bound=min_bound,
                                            max_bound=max_bound, factor=factor,
                                            offset=offset, X=X)
    
    s = z3.Solver()
    s.add(forest_con, min_con, max_con)
    s.add(z3.Or(z3.Sum(*outputs) < min_threshold,
                z3.Sum(*outputs) > max_threshold))
    res = s.check()
    return True if res == z3.unsat else False if res == z3.sat else None

if __name__ == "__main__":
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ], dtype = np.float64)
    np.random.seed(12345)
    y = np.array([0, 0.5, 1, 2], dtype = np.float64)
    forest = ensemble.RandomForestRegressor(n_estimators=3)
    forest.fit(X, y)
    cons, outputs = forest_constraint(forest)

    print(min_query(forest, [0, 0], [1, 1], 0))
    print(max_query(forest, [0, 0], [1, 1], 3))
    print(query(forest, [0, 0], [1, 1], 0, 2))
