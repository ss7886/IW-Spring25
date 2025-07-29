from typing import Iterable

import dtree

import sklearn.ensemble
import numpy as np

class ForestClassifier:
    trees: list[list[dtree.Tree]]  # trees[class_id][tree_id]
    n_class: int
    classes: list[float]
    n_trees: list[int]
    dim: int
    min_bound: np.typing.NDArray
    max_bound: np.typing.NDArray
    champ_min: list[float]
    champ_min_x: list[Iterable[float]]
    champ_max: list[float]
    champ_max_x: list[Iterable[float]]
    importances: list[list[np.typing.NDArray]]
    _gb: bool
    _factor: float
    _offset: np.typing.NDArray

    def __init__(self, trees: list[list[dtree.Tree]], classes: list[float] = None,
                 factor: float = None, offset: np.typing.NDArray = None,
                 gb: bool = False):
        if classes is None:
            classes = list(range(len(trees)))
        assert len(trees) == len(classes)
        self.trees = trees
        self.classes = np.array(classes)
        self.n_class = len(classes)
        self.n_trees = [len(class_trees) for class_trees in self.trees]
        self.dim = trees[0][0].dim
        for class_trees in trees:
            for tree in class_trees:
                assert tree.dim == self.dim

        self._factor = 1 / self.n_trees[0] if factor is None else factor
        self._offset = np.zeros(self.n_class) if offset is None else offset
        self._gb = gb

        self.min_bound = np.zeros(self.n_class)
        self.max_bound = np.zeros(self.n_class)
        self.importances = [None for _ in range(self.n_class)]
        self._update_stats()

        self.champ_min = [None for _ in range(self.n_class)]
        self.champ_min_x = [None for _ in range(self.n_class)]
        self.champ_max = [None for _ in range(self.n_class)]
        self.champ_max_x = [None for _ in range(self.n_class)]
    
    def __getitem__(self, key) -> dtree.Tree:
        return self.trees[key]
    
    def _update_stats(self, class_id: int = None, reset_importances=True):
        if class_id is not None:
            class_min = sum([tree.min for tree in self.trees[class_id]])
            self.min_bound[class_id] = class_min * self._factor + self._offset[class_id]

            class_max = sum([tree.max for tree in self.trees[class_id]])
            self.max_bound[class_id] = class_max * self._factor + self._offset[class_id]
        else:
            mins = np.array([[tree.min for tree in class_trees] for class_trees in self.trees])
            self.min_bound = np.sum(mins, axis=1) * self._factor
            self.min_bound += self._offset
            maxs = np.array([[tree.max for tree in class_trees] for class_trees in self.trees])
            self.max_bound = np.sum(maxs, axis=1) * self._factor
            self.max_bound += self._offset

        if reset_importances:
            if class_id is not None:
                self.importances[class_id] = None
            else:
                self.importances = [None for _ in range(self.n_class)]
    
    def free(self):
        for class_trees in self.trees:
            for tree in class_trees:
                tree.free()
    
    def copy(self) -> 'ForestClassifier':
        trees = []
        for class_trees in self.trees:
            new_class_trees = []
            for tree in class_trees:
                new_class_trees.append(tree.copy())
            trees.append(new_class_trees)
        
        forest = ForestClassifier(trees, self.classes.copy(), self._factor,
                                  self._offset.copy(), self._gb)
        forest.champ_min = self.champ_min
        forest.champ_min_x = self.champ_min_x
        forest.champ_max = self.champ_max
        forest.champ_max_x = self.champ_max_x
        return forest
    
    def avg_size_all(self):
        return sum([self.avg_size(i) for i in range(self.n_class)]) / self.n_class
    
    def avg_size(self, class_id):
        return sum([tree.size for tree in self.trees[class_id]]) / self.n_trees[class_id]
    
    def avg_depth_all(self):
        return sum([self.avg_depth(i) for i in range(self.n_class)]) / self.n_class
    
    def avg_depth(self, class_id):
        return sum([tree.depth for tree in self.trees[class_id]]) / self.n_trees[class_id]
    
    def print_summary(self):
        print(f"# classes: {self.n_class}")
        print(f"Size of forests: {self.n_trees}")
        print(f"Average Tree Size: {self.avg_size_all()}")
        print(f"Avg Max Depth: {self.avg_depth_all()}")
        for i in range(self.n_class):
            print(f"Class {self.classes[i]}:")
            if self._gb:
                print(f"Minimum (raw): [{self.min_bound[i]}, {self.champ_min[i]}]")
                print(f"Maximum (raw): [{self.champ_max[i]}, {self.max_bound[i]}]")
            else:
                print(f"Minimum: [{self.min_bound[i]}, {self.champ_min[i]}]")
                print(f"Maximum: [{self.champ_max[i]}, {self.max_bound[i]}]")
    
    def eval(self, x: Iterable[float], probs: bool = False,
             raw: bool = False, class_ids: Iterable[int] = None) -> float:
        if class_ids is None:
            class_ids = range(self.n_class)

        c_arr, n = dtree.Tree._get_c_arr(x)

        sum = np.zeros((n, len(class_ids)))

        for i, class_id in enumerate(class_ids):
            for tree in self.trees[class_id]:
                sum[:, i] += tree._eval_matrix(c_arr, n)

            sum[:, i] *= self._factor
            sum[:, i] += self._offset[class_id]

            max_index = np.argmax(sum[:, i])
            min_index = np.argmin(sum[:, i])
            if self.champ_max[class_id] is None or sum[max_index, i] > self.champ_max[class_id]:
                self.champ_max[class_id] = sum[max_index, i]
                self.champ_max_x[class_id] = x[max_index]
            if self.champ_min[class_id] is None or sum[min_index, i] < self.champ_min[class_id]:
                self.champ_min[class_id] = sum[min_index, i]
                self.champ_min_x[class_id] = x[min_index]

        if self._gb and not raw:
            if self.n_class == 2:
                # Apply expit function
                sum = 1 / (1 + np.exp(-1 * sum))
            else:
                # Apply softmax
                row_sums = np.sum(np.exp(sum), axis=1)
                sum = np.exp(sum) / np.repeat(row_sums, len(class_ids)).reshape(-1, len(class_ids))
        
        if probs:
            return sum
        else:
            return self.classes[np.argmax(sum, 1)]
    
    def sample(self, low: Iterable[float], high: Iterable[float], n: int):
        assert n >= 1
        assert len(low) == self.dim
        assert len(high) == self.dim
        X = np.random.uniform(low, high, (n, self.dim))
        self.eval(X)
    
    def prune_left(self, class_id: int, axis: int, threshold: float,
                   update_stats: bool = True) -> 'ForestClassifier':
        if self.importances[class_id] is None:
            self.importances[class_id] = self.feature_importance(class_id)
        for i, tree in enumerate(self.trees[class_id]):
            if self.importances[class_id][i][axis] > 0:
                tree.prune_left(axis, threshold)
                self.importances[class_id][i] = tree.feature_importance()
        if update_stats:
            self._update_stats(class_id=class_id)
            self.champ_min_x[class_id] = None
            self.champ_min[class_id] = None
            self.champ_max_x[class_id] = None
            self.champ_max[class_id] = None
        return self
    
    def prune_right(self, class_id: int, axis: int, threshold: float,
                    update_stats: bool = True) -> 'ForestClassifier':
        if self.importances[class_id] is None:
            self.importances[class_id] = self.feature_importance(class_id)
        for i, tree in enumerate(self.trees[class_id]):
            if self.importances[class_id][i][axis] > 0:
                tree.prune_right(axis, threshold)
                self.importances[class_id][i] = tree.feature_importance()
        if update_stats:
            self._update_stats(class_id=class_id)
            self.champ_min_x[class_id] = None
            self.champ_min[class_id] = None
            self.champ_max_x[class_id] = None
            self.champ_max[class_id] = None
        return self
    
    def prune_box(self, min_bound: Iterable[float], max_bound: Iterable[float],
                  class_ids: Iterable[int] = None):
        assert len(min_bound) == self.dim
        assert len(max_bound) == self.dim

        if class_ids is None:
            class_ids = range(self.n_class)

        for class_id in class_ids:
            for tree in self.trees[class_id]:
                tree.prune_box(min_bound, max_bound)
            self._update_stats(class_id=class_id)
            self.champ_min_x[class_id] = None
            self.champ_min[class_id] = None
            self.champ_max_x[class_id] = None
            self.champ_max[class_id] = None
        return self
    
    def prune_box_copy(self, min_bound: Iterable[float],
                       max_bound: Iterable[float]):
        assert len(min_bound) == self.dim
        assert len(max_bound) == self.dim

        trees = []
        for class_id in range(self.n_class):
            class_trees = []
            for tree in self.trees[class_id]:
                class_trees.append(tree.prune_box_copy(min_bound, max_bound))
            trees.append(class_trees)


        return ForestClassifier(trees, self.classes.copy(), self._factor,
                                self._offset.copy(), self._gb)
    
    def feature_importance(self, class_id: int) -> list[np.typing.NDArray]:
        importances = []
        for tree in self.trees[class_id]:
            importances.append(tree.feature_importance())
        return np.array(importances)
    
    def merge(self, class_id: int, n: int) -> 'ForestClassifier':
        importances = self.importances[class_id]
        if importances is None:
            importances = self.feature_importance(class_id)
        
        while self.n_trees[class_id] > n:
            champ_score = float("-inf")
            champ_indices = -1, -1
            for i in range(self.n_trees[class_id]):
                for j in range(i + 1, self.n_trees[class_id]):
                    corr = np.dot(importances[i], importances[j])
                    score = corr / (self.trees[class_id][i].size * self.trees[class_id][j].size)
                    if score > champ_score:
                        champ_score = score
                        champ_indices = i, j
            
            i, j = champ_indices
            new_tree = dtree.merge_trees(self.trees[class_id][i], self.trees[class_id][j])
            self.trees[class_id].pop(j).free()
            self.trees[class_id].pop(i).free()
            self.trees[class_id].append(new_tree)
            np.delete(importances, [i, j], axis=0)
            importances = np.vstack((importances,
                                    new_tree.feature_importance()))
            self.n_trees[class_id] -= 1
        
        self.importances[class_id] = importances
        self._update_stats(class_id=class_id, reset_importances=False)
        return self
    
    def merge_min(self, class_id: int, n: int, x: Iterable[float],
                  offset: float = 0.0) -> 'ForestClassifier':
        importances = self.importances[class_id]
        if importances is None:
            importances = self.feature_importance(class_id)
        evals = [tree.eval(x) for tree in self.trees[class_id]]
        offsets = [offset for tree in self.trees[class_id]]
        
        while self.n_trees[class_id] > n:
            champ_score = float("-inf")
            champ_indices = -1, -1
            for i in range(self.n_trees[class_id]):
                for j in range(i + 1, self.n_trees[class_id]):
                    corr = np.dot(importances[i], importances[j])
                    score = corr / (self.trees[class_id][i].size * self.trees[class_id][j].size)
                    if score > champ_score:
                        champ_score = score
                        champ_indices = i, j
            
            i, j = champ_indices
            tree_sum = evals[i] + offsets[i] + evals[j] + offsets[j]
            new_tree = dtree.merge_trees_min(self.trees[class_id][i], self.trees[class_id][j],
                                             tree_sum)
            self.trees[class_id].pop(j).free()
            self.trees[class_id].pop(i).free()
            self.trees[class_id].append(new_tree)
            np.delete(importances, [i, j], axis=0)
            importances = np.vstack((importances,
                                    new_tree.feature_importance()))
            evals.pop(j)
            evals.pop(i)
            evals.append(new_tree.eval(x))
            offsets.append(offsets.pop(j) + offsets.pop(i))
            self.n_trees[class_id] -= 1
        
        self.importances[class_id] = importances
        self._update_stats(class_id=class_id, reset_importances=False)
        return self
    
    def merge_max(self, class_id: int, n: int, x: Iterable[float],
                  offset: float = 0.0) -> 'ForestClassifier':
        importances = self.importances[class_id]
        if importances is None:
            importances = self.feature_importance(class_id)
        evals = [tree.eval(x) for tree in self.trees[class_id]]
        offsets = [offset for tree in self.trees[class_id]]
        
        while self.n_trees[class_id] > n:
            champ_score = float("-inf")
            champ_indices = -1, -1
            for i in range(self.n_trees[class_id]):
                for j in range(i + 1, self.n_trees[class_id]):
                    corr = np.dot(importances[i], importances[j])
                    score = corr / (self.trees[class_id][i].size * self.trees[class_id][j].size)
                    if score > champ_score:
                        champ_score = score
                        champ_indices = i, j
            
            i, j = champ_indices
            tree_sum = evals[i] + offsets[i] + evals[j] + offsets[j]
            new_tree = dtree.merge_trees_max(self.trees[class_id][i], self.trees[class_id][j],
                                             tree_sum)
            self.trees[class_id].pop(j).free()
            self.trees[class_id].pop(i).free()
            self.trees[class_id].append(new_tree)
            np.delete(importances, [i, j], axis=0)
            importances = np.vstack((importances,
                                    new_tree.feature_importance()))
            evals.pop(j)
            evals.pop(i)
            evals.append(new_tree.eval(x))
            offsets.append(offsets.pop(j) + offsets.pop(i))
            self.n_trees[class_id] -= 1
        
        self.importances[class_id] = importances
        self._update_stats(class_id=class_id, reset_importances=False)
        return self

def make_forest_classifier_sklearn(forest: sklearn.ensemble,
                                   gb: bool = False) -> ForestClassifier:
    class_trees = []
    classes = forest.classes_
    for i in range(len(classes)):
        trees = []
        for tree in forest.estimators_:
            if gb:
                if len(classes) == 2:
                    tree = tree[0]
                    factor = -1 if i == 0 else 1
                    trees.append(dtree.make_tree_sklearn(tree, 0, factor))
                else:
                    tree = tree[i]
                    trees.append(dtree.make_tree_sklearn(tree, 0))
            else:
                trees.append(dtree.make_tree_sklearn(tree, i))
        class_trees.append(trees)
    if gb:
        factor = forest.learning_rate
        if len(classes) == 2:
            offset = np.array([-1, 1]) * forest.init_.class_prior_[1]
        else:
            offset = forest.init_.class_prior_
    else:
        factor = 1 / forest.n_estimators
        offset = np.zeros(len(classes))
    return ForestClassifier(class_trees, classes, factor=factor, offset=offset, gb=gb)

if __name__ == "__main__":
    forest = ForestClassifier([[dtree.make_leaf(3, 0.25)], [dtree.make_leaf(3, 0.75)]])
    forest.sample([-1.0, -10.0, 5.0], [1.0, 10.0, 15.0], 10)
