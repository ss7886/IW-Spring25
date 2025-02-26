from typing import Iterable, TYPE_CHECKING
from weakref import finalize

from _tree_cffi import ffi, lib

import numpy as np
from sklearn.tree import DecisionTreeRegressor

if TYPE_CHECKING:
    from _cffi_backend import _CDataBase
    TreePtr = _CDataBase  # For type-checking
else:
    TreePtr = object

class treeCFFIError(Exception):
    def __init__(self, message):
        self.message = "_tree_cffi: " + message
        super().__init__(self.message)

class Tree:
    _tree: TreePtr
    left: 'Tree'
    right: 'Tree'
    min: float
    max: float
    depth: int
    size: int

    def __init__(self, ptr: TreePtr, left: 'Tree' = None, right: 'Tree' = None):
        self._tree = ptr
        self.dim = ptr[0].dim
        self.left = left
        self.right = right
        self.min = lib.treeMin(ptr)
        self.max = lib.treeMax(ptr)
        self.depth = lib.treeDepth(ptr)
        self.size = lib.treeSize(ptr)
    
    def _update_stats(self):
        if self._tree[0].split == ffi.NULL:
            self.left = None
            self.right = None
        else:
            if self.left is not None and (self._tree[0].split[0].left !=
                                          self.left._tree):
                self.left = None
            if self.right is not None and (self._tree[0].split[0].right !=
                                           self.right._tree):
                self.right = None

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
    
    def eval(self, x: Iterable[float]) -> float:
        assert self._check_tree()

        if isinstance(x, np.ndarray) and x.dtype == np.float64:
            c_arr = ffi.from_buffer("double[]", x)
        else:
            c_arr = ffi.new("double[]", list(x))
        
        return lib.treeEval(self._tree, c_arr)
    
    def prune_left(self, axis: int, threshold: float) -> 'Tree':
        assert self._check_tree()
        self._tree = lib.treePruneLeftInPlace(self._tree, axis, threshold)
        self._update_stats()
        return self

    def prune_right(self, axis: int, threshold: float) -> 'Tree':
        assert self._check_tree()
        self._tree = lib.treePruneRightInPlace(self._tree, axis, threshold)
        self._update_stats()
        return self

    def feature_importance(self) -> np.typing.NDArray:
        assert self._check_tree()

        importances = ffi.new(f"double[{self.dim}]")
        lib.featureImportance(importances, self._tree)
        buffer = ffi.buffer(importances, self.dim * ffi.sizeof("double"))
        arr = np.frombuffer(buffer, dtype=np.float64)

        # Make sure C Array gets freed after NumPy array is garbage collected
        finalize(arr, ffi.release, importances)

        return arr

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
    return Tree(ptr, left, right)

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

    tree1.free()
    tree2.free()
    merge.free()
