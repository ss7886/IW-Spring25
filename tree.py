from typing import Iterable, TYPE_CHECKING

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
    def __init__(self, ptr: TreePtr, left: 'Tree' = None, right: 'Tree' = None):
        self._tree = ptr
        self.is_leaf = ptr[0].isLeaf
        if (self.is_leaf):
            self.value = ptr[0].val
        else:
            self.axis = ptr[0].split[0].axis
            self.loc = ptr[0].split[0].loc
        self.left = left
        self.right = right
        self.min = lib.treeMin(ptr)
        self.max = lib.treeMax(ptr)
        self.depth = lib.treeDepth(ptr)
        self.size = lib.treeSize(ptr)
    
    def free(self):
        lib.freeTree(self._tree)
    
    def eval(self, x: Iterable[float]) -> float:
        if isinstance(x, np.ndarray) and x.dtype == np.float64:
            c_arr = ffi.from_buffer("double[]", x)
        else:
            c_arr = ffi.new("double[]", list(x))
        
        return lib.treeEval(self._tree, c_arr)

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
            return _make_leaf_ptr(dim, values[id][0][0])

        left = make_node(children_left[id])
        right = make_node(children_right[id])

        return _make_split_ptr(dim, feature[id], threshold[id], left, right)
    
    return Tree(make_node(0))

def merge_trees(tree1: Tree, tree2: Tree):
    res_ptr = ffi.new("Tree_T *")
    if not lib.treeMerge(res_ptr, tree1._tree, tree2._tree):
        raise treeCFFIError("treeMerge failed.")
    return Tree(res_ptr[0])

if __name__ == "__main__":
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

    tree1.free()
    tree2.free()
    merge.free()
