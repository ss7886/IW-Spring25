from typing import Iterable

import dtree

import sklearn.ensemble
import numpy as np

class Forest:
    trees: list[dtree.Tree]
    min: float
    max: float
    champ_min: float
    champ_min_x: Iterable[float]
    champ_max: float
    champ_max_x: Iterable[float]
    _factor: float

    def __init__(self, trees: list[dtree.Tree], factor: float = None):
        self.trees = trees
        self.n_trees = len(trees)        
        self._factor = 1 / self.n_trees if factor is None else factor
        self.min_bound = sum([tree.min for tree in trees]) * self._factor
        self.max_bound = sum([tree.max for tree in trees]) * self._factor
        self.champ_min = None
        self.champ_min_x = None
        self.champ_max = None
        self.champ_max_x = None
        self.importances = None
    
    def _update_stats(self, reset_importances=True):
        self.min_bound = sum([tree.min for tree in self.trees]) * self._factor
        self.max_bound = sum([tree.max for tree in self.trees]) * self._factor
        if reset_importances:
            self.importances = None
    
    def free(self):
        for tree in self.trees:
            tree.free()
    
    def copy(self) -> 'Forest':
        trees = []
        for tree in self.trees:
            trees.append(tree.copy())
        return Forest(trees, self._factor)
    
    def print_summary(self):
        print(f"Size of forest: {self.n_trees}")
        avg_size = sum([tree.size for tree in self.trees]) / self.n_trees
        print(f"Average Tree Size: {avg_size}")
        avg_depth = sum([tree.depth for tree in self.trees]) / self.n_trees
        print(f"Avg Max Depth: {avg_depth}")
        print(f"Minimum: [{self.min_bound}, {self.champ_min}]")
        print(f"Maximum: [{self.champ_max}, {self.max_bound}]")
    
    def eval(self, x: Iterable[float]) -> float:
        sum = 0.0
        for tree in self.trees:
            sum += tree.eval(x)
        sum *= self._factor
        if self.champ_max is None or sum > self.champ_max:
            self.champ_max = sum
            self.champ_max_x = x
        if self.champ_min is None or sum < self.champ_min:
            self.champ_min = sum
            self.champ_min_x = x
        return sum
    
    def prune_left(self, axis: int, threshold: float) -> 'Forest':
        for tree in self.trees:
            tree.prune_left(axis, threshold)
        self._update_stats()
        if self.champ_min_x[axis] > threshold:
            self.champ_min_x = None
            self.champ_min = None
        if self.champ_max_x[axis] > threshold:
            self.champ_max_x = None
            self.champ_max = None
        return self
    
    def prune_right(self, axis: int, threshold: float) -> 'Forest':
        for tree in self.trees:
            tree.prune_right(axis, threshold)
        self._update_stats()
        if self.champ_min_x[axis] < threshold:
            self.champ_min_x = None
            self.champ_min = None
        if self.champ_max_x[axis] < threshold:
            self.champ_max_x = None
            self.champ_max = None
        return self
    
    def feature_importance(self) -> list[np.typing.NDArray]:
        importances = []
        for tree in self.trees:
            importances.append(tree.feature_importance())
        return importances
    
    def merge(self, n: int) -> 'Forest':
        if self.importances is None:
            self.importances = self.feature_importance()
        
        while self.n_trees > n:
            champ_score = float("-inf")
            champ_indices = -1, -1
            for i in range(self.n_trees):
                for j in range(i + 1, self.n_trees):
                    corr = np.dot(self.importances[i], self.importances[j])
                    score = corr / (self.trees[i].size * self.trees[j].size)
                    if score > champ_score:
                        champ_score = score
                        champ_indices = i, j
            
            i, j = champ_indices
            new_tree = dtree.merge_trees(self.trees[i], self.trees[j])
            self.trees.pop(j).free()
            self.trees.pop(i).free()
            self.trees.append(new_tree)
            self.importances.pop(j)
            self.importances.pop(i)
            self.importances.append(new_tree.feature_importance())
            self.n_trees -= 1
        
        self._update_stats(False)
        return self

def make_forest_sklearn(forest: sklearn.ensemble, **kwargs):
    trees = []
    for tree in forest.estimators_:
        trees.append(dtree.make_tree_sklearn(tree, **kwargs))
    return Forest(trees)
