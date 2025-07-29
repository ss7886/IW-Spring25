from typing import Iterable

import dtree

import sklearn.ensemble
import numpy as np

class Forest:
    trees: list[dtree.Tree]
    n_trees: int
    dim: int
    min_bound: float
    max_bound: float
    champ_min: float
    champ_min_x: Iterable[float]
    champ_max: float
    champ_max_x: Iterable[float]
    importances: list[np.typing.NDArray]
    _factor: float
    _offset: float

    def __init__(self, trees: list[dtree.Tree], factor: float = None,
                 offset: float = 0):
        assert len(trees) >= 1
        self.trees = trees
        self.n_trees = len(trees)
        self.dim = trees[0].dim
        for tree in trees:
            assert tree.dim == self.dim

        self._factor = 1 / self.n_trees if factor is None else factor
        self._offset = offset
        self.min_bound = sum([tree.min for tree in self.trees]) * self._factor
        self.min_bound += self._offset
        self.max_bound = sum([tree.max for tree in self.trees]) * self._factor 
        self.max_bound += self._offset
        self.champ_min = None
        self.champ_min_x = None
        self.champ_max = None
        self.champ_max_x = None
        self.importances = None
    
    def __getitem__(self, key) -> dtree.Tree:
        return self.trees[key]
    
    def _update_stats(self, reset_importances=True):
        self.min_bound = sum([tree.min for tree in self.trees]) * self._factor
        self.min_bound += self._offset
        self.max_bound = sum([tree.max for tree in self.trees]) * self._factor 
        self.max_bound += self._offset
        if reset_importances:
            self.importances = None
    
    def free(self):
        for tree in self.trees:
            tree.free()
    
    def copy(self) -> 'Forest':
        trees = []
        for tree in self.trees:
            trees.append(tree.copy())

        forest = Forest(trees, self._factor, self._offset)
        forest.champ_min = self.champ_min
        forest.champ_min_x = self.champ_min_x
        forest.champ_max = self.champ_max
        forest.champ_max_x = self.champ_max_x
        return forest
    
    def avg_size(self):
        return sum([tree.size for tree in self.trees]) / self.n_trees
    
    def avg_depth(self):
        return sum([tree.depth for tree in self.trees]) / self.n_trees
    
    def print_summary(self):
        print(f"Size of forest: {self.n_trees}")
        print(f"Average Tree Size: {self.avg_size()}")
        print(f"Avg Max Depth: {self.avg_depth()}")
        print(f"Minimum: [{self.min_bound}, {self.champ_min}]")
        print(f"Maximum: [{self.champ_max}, {self.max_bound}]")
    
    def eval(self, x: Iterable[float]) -> float:
        c_arr, n = dtree.Tree._get_c_arr(x)

        if n == 1:
            sum = 0.0
        else:
            sum = np.zeros(n)

        for tree in self.trees:
            if n == 1:
                sum += tree._eval(c_arr)
            else:
                sum += tree._eval_matrix(c_arr, n)

        sum *= self._factor
        sum += self._offset

        if n == 1:
            if self.champ_max is None or sum > self.champ_max:
                self.champ_max = sum
                self.champ_max_x = x
            if self.champ_min is None or sum < self.champ_min:
                self.champ_min = sum
                self.champ_min_x = x
        else:
            max_index = np.argmax(sum)
            min_index = np.argmin(sum)
            if self.champ_max is None or sum[max_index] > self.champ_max:
                self.champ_max = sum[max_index]
                self.champ_max_x = x[max_index]
            if self.champ_min is None or sum[min_index] < self.champ_min:
                self.champ_min = sum[min_index]
                self.champ_min_x = x[min_index]
        return sum
    
    def sample(self, low: Iterable[float], high: Iterable[float], n: int):
        assert n >= 1
        assert len(low) == self.dim
        assert len(high) == self.dim
        X = np.random.uniform(low, high, (n, self.dim))
        self.eval(X)
    
    def prune_left(self, axis: int, threshold: float) -> 'Forest':
        for tree in self.trees:
            tree.prune_left(axis, threshold)
        self._update_stats()
        if self.champ_min_x is not None and self.champ_min_x[axis] > threshold:
            self.champ_min_x = None
            self.champ_min = None
        if self.champ_max_x is not None and self.champ_max_x[axis] > threshold:
            self.champ_max_x = None
            self.champ_max = None
        return self
    
    def prune_right(self, axis: int, threshold: float) -> 'Forest':
        for tree in self.trees:
            tree.prune_right(axis, threshold)
        self._update_stats()
        if self.champ_min_x is not None and self.champ_min_x[axis] < threshold:
            self.champ_min_x = None
            self.champ_min = None
        if self.champ_max_x is not None and self.champ_max_x[axis] < threshold:
            self.champ_max_x = None
            self.champ_max = None
        return self
    
    def prune_box(self, min_bound: Iterable[float],
                  max_bound: Iterable[float]):
        assert len(min_bound) == self.dim
        assert len(max_bound) == self.dim

        for tree in self.trees:
            tree.prune_box(min_bound, max_bound)
        self._update_stats()
        self.champ_min = None
        self.champ_min_x = None
        self.champ_max = None
        self.champ_max_x = None
        return self
    
    def prune_box_copy(self, min_bound: Iterable[float],
                       max_bound: Iterable[float]):
        assert len(min_bound) == self.dim
        assert len(max_bound) == self.dim

        trees = []
        for tree in self.trees:
            trees.append(tree.prune_box_copy(min_bound, max_bound))
        return Forest(trees, self._factor, self._offset)
    
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
        
        self._update_stats(reset_importances=False)
        return self
    
    def merge_min(self, n: int, x: Iterable[float],
                  offset: float = 0.0) -> 'Forest':
        self.eval(x)
        if self.importances is None:
            self.importances = self.feature_importance()
        evals = [tree.eval(x) for tree in self.trees]
        offsets = [offset for tree in self.trees]
        
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
            tree_sum = evals[i] + offsets[i] + evals[j] + offsets[j]
            new_tree = dtree.merge_trees_min(self.trees[i], self.trees[j],
                                             tree_sum)
            self.trees.pop(j).free()
            self.trees.pop(i).free()
            self.trees.append(new_tree)
            self.importances.pop(j)
            self.importances.pop(i)
            self.importances.append(new_tree.feature_importance())
            evals.pop(j)
            evals.pop(i)
            evals.append(new_tree.eval(x))
            offsets.append(offsets.pop(j) + offsets.pop(i))
            self.n_trees -= 1
        
        self._update_stats(reset_importances=False)
        return self
    
    def merge_max(self, n: int, x: Iterable[float],
                  offset: float = 0.0) -> 'Forest':
        self.eval(x)
        if self.importances is None:
            self.importances = self.feature_importance()
        evals = [tree.eval(x) for tree in self.trees]
        offsets = [offset for tree in self.trees]
        
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
            tree_sum = evals[i] + offsets[i] + evals[j] + offsets[j]
            new_tree = dtree.merge_trees_max(self.trees[i], self.trees[j],
                                             tree_sum)
            self.trees.pop(j).free()
            self.trees.pop(i).free()
            self.trees.append(new_tree)
            self.importances.pop(j)
            self.importances.pop(i)
            self.importances.append(new_tree.feature_importance())
            evals.pop(j)
            evals.pop(i)
            evals.append(new_tree.eval(x))
            offsets.append(offsets.pop(j) + offsets.pop(i))
            self.n_trees -= 1
        
        self._update_stats(reset_importances=False)
        return self

def make_forest_sklearn(forest: sklearn.ensemble, gb: bool = False,
                        **kwargs) -> Forest:
    trees = []
    for tree in forest.estimators_:
        if isinstance(tree, np.ndarray):
            tree = tree[0]
        trees.append(dtree.make_tree_sklearn(tree, **kwargs))
    if gb:
        factor = forest.learning_rate
        offset = forest.init_.constant_[0, 0]
    else:
        factor = None
        offset = 0
    return Forest(trees, factor=factor, offset=offset)

if __name__ == "__main__":
    forest = Forest([dtree.make_leaf(3, 1)])
    forest.sample([-1.0, -10.0, 5.0], [1.0, 10.0, 15.0], 10)
