#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "tree.h"

double treeMin(Tree_T tree) {
    assert(tree != NULL);

    if(tree->isLeaf) {
        return tree->val;
    }

    assert(tree->split != NULL);
    return tree->split->min;
}

double treeMax(Tree_T tree) {
    assert(tree != NULL);

    if(tree->isLeaf) {
        return tree->val;
    }

    assert(tree->split != NULL);
    return tree->split->max;
}

uint32_t treeDepth(Tree_T tree) {
    assert(tree != NULL);

    if(tree->isLeaf) {
        return 0;
    }

    assert(tree->split != NULL);
    return tree->split->depth;
}

uint32_t treeSize(Tree_T tree) {
    assert(tree != NULL);

    if(tree->isLeaf) {
        return 1;
    }

    assert(tree->split != NULL);
    return tree->split->size;
}

bool validTree(Tree_T tree) {
    assert(tree != NULL);

    return validLeaf(tree) || validSplit(tree);
}

bool validLeaf(Tree_T tree) {
    assert(tree != NULL);

    return tree->isLeaf && tree->split == NULL;
}

bool validSplit(Tree_T tree) {
    assert(tree != NULL && tree->split != NULL);
    Split_T split = tree->split;

    assert(split->left != NULL && split->right != NULL);
    Tree_T left = split->left;
    Tree_T right = split->right;

    return !tree->isLeaf &&
    tree->dim == left->dim &&
    tree->dim == right->dim &&
    split->axis < tree->dim &&
    split->min == fmin(treeMin(left), treeMin(right)) &&
    split->max == fmax(treeMax(left), treeMax(right)) &&
    split->depth == fmax(treeDepth(left), treeDepth(right)) + 1 &&
    split->size == treeSize(left) + treeSize(right) &&
    validTree(left) &&
    validTree(right);
}

bool makeLeaf(Tree_T * result, uint32_t dim, double val) {
    assert(result != NULL);
    *result = NULL;

    Tree_T newTree = malloc(sizeof(struct tree));
    if (newTree == NULL) {
        return false;
    }

    newTree->isLeaf = true;
    newTree->dim = dim;
    newTree->val = val;
    newTree->split = NULL;
    assert(validTree(newTree));

    *result = newTree;
    return true;
}

bool makeSplit(Tree_T * result, uint32_t dim, uint32_t axis,
               double loc, Tree_T left, Tree_T right) {
    assert(result != NULL);
    assert(axis < dim);
    assert(validTree(left) && validTree(right));
    assert(dim == left->dim && dim == right->dim);
    *result = NULL;

    Tree_T newTree = malloc(sizeof(struct tree));
    if(newTree == NULL) {
        return false;
    }

    Split_T newSplit = malloc(sizeof(struct split));
    if(newSplit == NULL) {
        free(newTree);
        return false;
    }

    newTree->isLeaf = false;
    newTree->dim = dim;
    newTree->split = newSplit;
    newSplit->axis = axis;
    newSplit->loc = loc;
    newSplit->left = left;
    newSplit->right = right; 
    newSplit->min = fmin(treeMin(left), treeMin(right));
    newSplit->max = fmax(treeMax(left), treeMax(right));
    newSplit->depth = fmax(treeDepth(left), treeDepth(right)) + 1;
    newSplit->size = treeSize(left) + treeSize(right);
    assert(validTree(newTree));

    *result = newTree;
    return true;
}

bool copyTreeSafe(Tree_T * result, Tree_T tree) {
    *result = NULL;
    if (tree->isLeaf) {
        return makeLeaf(result, tree->dim, tree->val);
    }

    Split_T split = tree->split;
    Tree_T left, right;
    if (!copyTreeSafe(&left, split->left)) {
        return false;
    }
    if (!copyTreeSafe(&right, split->right)) {
        freeTree(left);
        return false;
    }
    return makeSplit(result, tree->dim, split->axis, split->loc, left, right);
}

bool copyTree(Tree_T * result, Tree_T tree) {
    /* Only validate arguments once. */
    assert(validTree(tree));
    assert(result != NULL);

    return copyTreeSafe(result, tree);
};

void freeTree(Tree_T tree) {
    if (tree == NULL) {
        return;
    }

    if (tree->split != NULL) {
        Split_T split = tree->split;
        if (split->left != NULL) {
            freeTree(split->left);
            split->left = NULL;
        }
        if (split->right != NULL) {
            freeTree(split->right);
            split->right = NULL;
        }
        free(split);
        tree->split = NULL;
    }
    free(tree);
    return;
}

/* See treeEval */
double treeEvalSafe(Tree_T tree, double x[]) {
    if(tree->isLeaf) {
        return tree->val;
    }

    Split_T split = tree->split;

    if (x[split->axis] <= split->loc) {
        return treeEvalSafe(split->left, x);
    } else {
        return treeEvalSafe(split->right, x);
    }
}

double treeEval(Tree_T tree, double x[]) {
    /* Only validate arguments once. */
    assert(validTree(tree));
    assert(x != NULL);

    return treeEvalSafe(tree, x);
};

bool treePruneLeftSafe(Tree_T * result, Tree_T tree, uint32_t axis, 
                       double loc) {
    *result = NULL;
    if(tree->isLeaf) {
        return copyTree(result, tree);
    }

    Split_T split = tree->split;
    bool success;
    if(axis != split->axis) {
        /* Prune both subtrees. */
        Tree_T left, right;
        if(!treePruneLeftSafe(&left, split->left, axis, loc)) {
            return false;
        }
        if(!treePruneLeftSafe(&right, split->right, axis, loc)) {
            freeTree(left);
            return false;
        }
        return makeSplit(result, tree->dim, split->axis,
                         split->loc, left, right);
    }

    if(loc <= split->loc) {
        /* Prune left subtree, drop right subtree. */
        return treePruneLeftSafe(result, split->left, axis, loc);
    } else {
        /* Copy left subtree, prune right subtree. */
        Tree_T left, right;
        if(!copyTreeSafe(&left, split->left)) {
            return false;
        }
        if(!treePruneLeftSafe(&right, split->right, axis, loc)) {
            freeTree(left);
            return false;
        }
        return makeSplit(result, tree->dim, split->axis,
                         split->loc, left, right);
    }
}

bool treePruneLeft(Tree_T * result, Tree_T tree, uint32_t axis, double loc) {
    /* Only validate arguments once. */
    assert(result != NULL);
    assert(validTree(tree));
    assert(axis < tree->dim);

    return treePruneLeftSafe(result, tree, axis, loc);
}

bool treePruneRightSafe(Tree_T * result, Tree_T tree, uint32_t axis, 
                       double loc) {
    *result = NULL;
    if(tree->isLeaf) {
        return copyTree(result, tree);
    }

    Split_T split = tree->split;
    bool success;
    if(axis != split->axis) {
        /* Prune both subtrees. */
        Tree_T left, right;
        if(!treePruneRightSafe(&left, split->left, axis, loc)) {
            return false;
        }
        if(!treePruneRightSafe(&right, split->right, axis, loc)) {
            freeTree(left);
            return false;
        }
        return makeSplit(result, tree->dim, split->axis,
                         split->loc, left, right);
    }

    if(loc <= split->loc) {
        /* Prune left subtree, copy right subtree. */
        Tree_T left, right;
        if(!treePruneRightSafe(&left, split->left, axis, loc)) {
            return false;
        }
        if(!copyTreeSafe(&right, split->right)) {
            freeTree(left);
            return false;
        }
        return makeSplit(result, tree->dim, split->axis,
            split->loc, left, right);
    } else {
        /* Prune right subtree, drop left subtree. */
        return treePruneRightSafe(result, split->right, axis, loc);
    }
}

bool treePruneRight(Tree_T * result, Tree_T tree, uint32_t axis, double loc) {
    /* Only validate arguments once. */
    assert(result != NULL);
    assert(validTree(tree));
    assert(axis < tree->dim);

    return treePruneRightSafe(result, tree, axis, loc);
}

