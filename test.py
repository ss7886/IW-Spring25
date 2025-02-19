import ctypes
import os

import numpy as np

# Load the shared library based on OS
if os.name == "nt":  # Windows
    tree_lib = ctypes.CDLL("./tree.dll")
else:  # Unix-based systems
    tree_lib = ctypes.CDLL("./tree.so")

# Forward declare the structures
class Tree(ctypes.Structure):
    pass

class Split(ctypes.Structure):
    pass

# Define Tree and Split structures
Tree_T = ctypes.POINTER(Tree)
Split_T = ctypes.POINTER(Split)

Tree._fields_ = [
    ("isLeaf", ctypes.c_bool),
    ("dim", ctypes.c_uint32),
    ("val", ctypes.c_double),
    ("split", Split_T)
]

Split._fields_ = [
    ("axis", ctypes.c_uint32),
    ("loc", ctypes.c_double),
    ("left", Tree_T),
    ("right", Tree_T),
    ("min", ctypes.c_double),
    ("max", ctypes.c_double),
    ("depth", ctypes.c_uint32),
    ("size", ctypes.c_uint32)
]

# Function bindings
tree_lib.treeMin.argtypes = [Tree_T]
tree_lib.treeMin.restype = ctypes.c_double

tree_lib.treeMax.argtypes = [Tree_T]
tree_lib.treeMax.restype = ctypes.c_double

tree_lib.treeDepth.argtypes = [Tree_T]
tree_lib.treeDepth.restype = ctypes.c_uint32

tree_lib.treeSize.argtypes = [Tree_T]
tree_lib.treeSize.restype = ctypes.c_uint32

tree_lib.validTree.argtypes = [Tree_T]
tree_lib.validTree.restype = ctypes.c_bool

tree_lib.validLeaf.argtypes = [Tree_T]
tree_lib.validLeaf.restype = ctypes.c_bool

tree_lib.validSplit.argtypes = [Tree_T]
tree_lib.validSplit.restype = ctypes.c_bool

tree_lib.makeLeaf.argtypes = [ctypes.POINTER(Tree_T), ctypes.c_uint32, ctypes.c_double]
tree_lib.makeLeaf.restype = ctypes.c_bool

tree_lib.makeSplit.argtypes = [
    ctypes.POINTER(Tree_T),
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_double,
    Tree_T,
    Tree_T
]
tree_lib.makeSplit.restype = ctypes.c_bool

tree_lib.freeTree.argtypes = [Tree_T]
tree_lib.freeTree.restype = None

tree_lib.treeEval.argtypes = [Tree_T, ctypes.POINTER(ctypes.c_double)]
tree_lib.treeEval.restype = ctypes.c_double

# Example usage (if necessary)
if __name__ == "__main__":
    leaf1, leaf2, tree = Tree_T(), Tree_T(), Tree_T()
    assert(tree_lib.makeLeaf(ctypes.byref(leaf1), 3, 5.0))
    assert(tree_lib.makeLeaf(ctypes.byref(leaf2), 3, -2.0))
    assert(tree_lib.makeSplit(ctypes.byref(tree), 3, 0, 0.0, leaf1, leaf2))
    print(f"Tree Height: {tree_lib.treeDepth(tree)}")
    print(f"Tree Size: {tree_lib.treeSize(tree)}")
    print(f"Min Value: {tree_lib.treeMin(tree)}")
    print(f"Max Value: {tree_lib.treeMax(tree)}")

    x1 = np.array([-1.0, 0.0, 0.0])
    x2 = np.array([1.0, 0.0, 0.0])

    y1 = tree_lib.treeEval(tree, x1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    y2 = tree_lib.treeEval(tree, x2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

    print(f"f({x1}) = {y1}")
    print(f"f({x2}) = {y2}")
    
    tree_lib.freeTree(tree)
