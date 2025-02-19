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

    struct tree * newTree = malloc(sizeof(struct tree));
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

    struct tree * newTree = malloc(sizeof(struct tree));
    if(newTree == NULL) {
        return false;
    }

    struct split * newSplit = malloc(sizeof(struct split));
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

void freeTree(Tree_T tree) {
    if (tree == NULL) {
        return;
    }

    if (tree->split != NULL) {
        Split_T split = tree->split;
        if (split->left != NULL) {
            freeTree(split->left);
        }
        if (split->right != NULL) {
            freeTree(split->right);
        }
        free(split);
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

    treeEvalSafe(tree, x);
};
