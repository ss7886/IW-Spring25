from typing import Iterable, Tuple, TYPE_CHECKING
from weakref import finalize

from _tree_cffi import ffi, lib

import numpy as np
from numpy.typing import NDArray
from sklearn.tree import DecisionTreeRegressor

if TYPE_CHECKING:
    from _cffi_backend import _CDataBase
    TreePtr = _CDataBase  # For type-checking
    ArrayPtr = _CDataBase
else:
    TreePtr = object
    ArrayPtr = object

class treeCFFIError(Exception):
    def __init__(self, message):
        self.message = "_tree_cffi: " + message
        super().__init__(self.message)

class Tree:
    _tree: TreePtr
    dim: int
    min: float
    max: float
    depth: int
    size: int

    def __init__(self, ptr: TreePtr):
        self._tree = ptr
        self.dim = lib.treeDim(ptr)
        self.min = lib.treeMin(ptr)
        self.max = lib.treeMax(ptr)
        self.depth = lib.treeDepth(ptr)
        self.size = lib.treeSize(ptr)
    
    def _update_stats(self):
        self.min = lib.treeMin(self._tree)
        self.max = lib.treeMax(self._tree)
        self.depth = lib.treeDepth(self._tree)
        self.size = lib.treeSize(self._tree)
    
    def _check_tree(self):
        if self._tree == ffi.NULL:
            raise treeCFFIError("Tree pointer is NULL.")
        return True
    
    def free(self):
        assert self._check_tree()
        lib.freeTree(self._tree)
        self._tree = ffi.NULL
    
    def copy(self) -> 'Tree':
        """
        Does not create python objects for tree.left and tree.right, even if they
        are defined in the original.
        """
        assert self._check_tree()

        pointer = ffi.new("Tree_T *")
        if not lib.copyTree(pointer, self._tree):
            raise treeCFFIError("copyTree failed.")
        return Tree(pointer[0])
    
    def _eval(self, c_arr: ArrayPtr) -> float:
        return lib.treeEval(self._tree, c_arr)
    
    def _eval_matrix(self, c_arr: ArrayPtr, n: int) -> NDArray:
        res = lib.treeEvalMatrix(self._tree, c_arr, n)
        if res == ffi.NULL:
            raise treeCFFIError("treeEvalMatrix failed.")
        buffer = ffi.buffer(res, n * ffi.sizeof("double"))
        arr = np.frombuffer(buffer, dtype=np.float64).copy()
        lib.freeArray(res)

        return arr
    
    @staticmethod
    def _get_c_arr(x: Iterable[float]) -> Tuple[ArrayPtr, int]:
        n = 1
        if isinstance(x, np.ndarray) and x.dtype == np.float64:
            if x.ndim > 1:
                n, _ = x.shape
            c_arr = ffi.from_buffer("double[]", x)
        else:
            c_arr = ffi.new("double[]", list(x))
            finalize(c_arr, ffi.release, c_arr)
        return c_arr, n
        
    def eval(self, x: Iterable[float]) -> float | NDArray:
        assert self._check_tree()
        c_arr, n = Tree._get_c_arr(x)
        if n == 1:
            return self._eval(c_arr)
        else:
            return self._eval_matrix(c_arr, n)
    
    def prune_left(self, axis: int, threshold: float) -> 'Tree':
        assert self._check_tree()
        self._tree = lib.treePruneLeftInPlace(self._tree, axis, threshold)
        self._update_stats()
        return self
    
    def prune_left_copy(self, axis: int, threshold: float) -> 'Tree':
        assert self._check_tree()
        treePtr = lib.treePruneLeft(self._tree, axis, threshold)
        return Tree(treePtr)

    def prune_right(self, axis: int, threshold: float) -> 'Tree':
        assert self._check_tree()
        self._tree = lib.treePruneRightInPlace(self._tree, axis, threshold)
        self._update_stats()
        return self
    
    def prune_right_copy(self, axis: int, threshold: float) -> 'Tree':
        assert self._check_tree()
        treePtr = lib.treePruneRight(self._tree, axis, threshold)
        return Tree(treePtr)
    
    def prune_box(self, min_bound: Iterable[float],
                  max_bound: Iterable[float]):
        assert self._check_tree()
        assert len(min_bound) == self.dim
        assert len(max_bound) == self.dim

        for i, x in enumerate(min_bound):
            self.prune_right(i, x)
        for i, x in enumerate(max_bound):
            self.prune_left(i, x)
        return self

    def feature_importance(self) -> NDArray:
        assert self._check_tree()

        importances = lib.featureImportance(self._tree)
        if importances == ffi.NULL:
            raise treeCFFIError("featureImportance failed.")
        buffer = ffi.buffer(importances, self.dim * ffi.sizeof("double"))
        arr = np.frombuffer(buffer, dtype=np.float64)

        # Make sure C Array gets freed after NumPy array is garbage collected
        finalize(arr, lib.freeArray, importances)

        return arr
    
    def find_min(self, min_bounds: NDArray = None,
                 max_bounds: NDArray = None) -> Tuple[NDArray, NDArray]:
        assert self._check_tree()

        if min_bounds is None:
            min_bounds = np.ones(self.dim) * -np.inf
        if max_bounds is None:
            max_bounds = np.ones(self.dim) * np.inf
        
        min_buffer = ffi.from_buffer("double[]", min_bounds)
        max_buffer = ffi.from_buffer("double[]", max_bounds)
        
        lib.findMin(self._tree, min_buffer, max_buffer)

        return min_bounds, max_bounds
    
    def find_max(self, min_bounds: NDArray = None,
                 max_bounds: NDArray = None) -> Tuple[NDArray, NDArray]:
        assert self._check_tree()

        if min_bounds is None:
            min_bounds = np.ones(self.dim) * -np.inf
        if max_bounds is None:
            max_bounds = np.ones(self.dim) * np.inf
        
        min_buffer = ffi.from_buffer("double[]", min_bounds)
        max_buffer = ffi.from_buffer("double[]", max_bounds)
        
        lib.findMax(self._tree, min_buffer, max_buffer)

        return min_bounds, max_bounds

def _make_leaf_ptr(dim: int, value: float) -> TreePtr:
    pointer = ffi.new("Tree_T *")
    if not lib.makeLeaf(pointer, dim, value):
        raise treeCFFIError("makeLeaf failed.")
    return pointer[0]

def make_leaf(dim: int, value: float) -> Tree:
    ptr = _make_leaf_ptr(dim, value)
    return Tree(ptr)

def _make_split_ptr(dim: int, axis: int, loc: float, left: TreePtr,
                    right: TreePtr) -> TreePtr:
    pointer = ffi.new("Tree_T *")
    if not lib.makeSplit(pointer, dim, axis, loc, left, right):
        raise treeCFFIError("makeSplit failed.")
    return pointer[0]

def make_split(dim: int, axis: int, loc: float, left: Tree,
               right: Tree) -> Tree:
    assert left._check_tree()
    assert right._check_tree()
    ptr = _make_split_ptr(dim, axis, loc, left._tree, right._tree)
    return Tree(ptr)

def make_tree_sklearn(reg: DecisionTreeRegressor) -> Tree:
    dim = reg.n_features_in_
    children_left = reg.tree_.children_left
    children_right = reg.tree_.children_right
    feature = reg.tree_.feature
    threshold = reg.tree_.threshold
    values = reg.tree_.value

    def make_node(id):
        if children_left[id] == children_right[id]:
            val = values[id][0][0]
            return _make_leaf_ptr(dim, val)

        left = make_node(children_left[id])
        right = make_node(children_right[id])
        loc = threshold[id]

        return _make_split_ptr(dim, feature[id], loc, left, right)
    
    return Tree(make_node(0))

def merge_trees(tree1: Tree, tree2: Tree) -> Tree:
    assert tree1._check_tree()
    assert tree2._check_tree()

    res_ptr = ffi.new("Tree_T *")
    if not lib.treeMerge(res_ptr, tree1._tree, tree2._tree):
        raise treeCFFIError("treeMerge failed.")
    return Tree(res_ptr[0])

def merge_trees_max(tree1: Tree, tree2: Tree, val: float) -> Tree:
    assert tree1._check_tree()
    assert tree2._check_tree()

    res_ptr = ffi.new("Tree_T *")
    if not lib.treeMergeMax(res_ptr, tree1._tree, tree2._tree, val):
        raise treeCFFIError("treeMergeMax failed.")
    return Tree(res_ptr[0])

def merge_trees_min(tree1: Tree, tree2: Tree, val: float) -> Tree:
    assert tree1._check_tree()
    assert tree2._check_tree()

    res_ptr = ffi.new("Tree_T *")
    if not lib.treeMergeMin(res_ptr, tree1._tree, tree2._tree, val):
        raise treeCFFIError("treeMergeMin failed.")
    return Tree(res_ptr[0])

if __name__ == "__main__":
    split_1a = make_split(3, 1, 0.25, make_leaf(2, 1), make_leaf(2, 2))
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
    print("Trees pass")

    tree1.free()
    tree2.free()
    merge.free()
