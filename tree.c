#include <assert.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "tree.h"

double treeMin(Tree_T tree) {
    assert(tree != NULL);

    if (tree->isLeaf) {
        return tree->val;
    }

    assert(tree->split != NULL);
    return tree->split->min;
}

double treeMax(Tree_T tree) {
    assert(tree != NULL);

    if (tree->isLeaf) {
        return tree->val;
    }

    assert(tree->split != NULL);
    return tree->split->max;
}

uint32_t treeDepth(Tree_T tree) {
    assert(tree != NULL);

    if (tree->isLeaf) {
        return 0;
    }

    assert(tree->split != NULL);
    return tree->split->depth;
}

uint32_t treeSize(Tree_T tree) {
    assert(tree != NULL);

    if (tree->isLeaf) {
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
    if (newTree == NULL) {
        return false;
    }

    Split_T newSplit = malloc(sizeof(struct split));
    if (newSplit == NULL) {
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
    if(!makeSplit(result, tree->dim, split->axis, split->loc, left, right)) {
        freeTrees(2, left, right);
        return false;
    }
    return true;
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

void freeTrees(int count, ...) {
    va_list args;
    va_start(args, count);

    int i = 0;
    for (i = 0; i < count; i++) {
        freeTree(va_arg(args, Tree_T));
    }

    va_end(args);
}

/* See treeEval */
double treeEvalSafe(Tree_T tree, double x[]) {
    if (tree->isLeaf) {
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
    if (tree->isLeaf) {
        return copyTree(result, tree);
    }

    Split_T split = tree->split;
    if (axis != split->axis) {
        /* Prune both subtrees. */
        Tree_T left, right;
        if (!treePruneLeftSafe(&left, split->left, axis, loc)) {
            return false;
        }
        if (!treePruneLeftSafe(&right, split->right, axis, loc)) {
            freeTree(left);
            return false;
        }
        if (!makeSplit(result, tree->dim, split->axis,
                         split->loc, left, right)) {
            freeTrees(2, left, right);
            return false;
        }
        return true;
    }

    if (loc <= split->loc) {
        /* Prune left subtree, drop right subtree. */
        return treePruneLeftSafe(result, split->left, axis, loc);
    } else {
        /* Copy left subtree, prune right subtree. */
        Tree_T left, right;
        if (!copyTreeSafe(&left, split->left)) {
            return false;
        }
        if (!treePruneLeftSafe(&right, split->right, axis, loc)) {
            freeTree(left);
            return false;
        }
        if (!makeSplit(result, tree->dim, split->axis,
                         split->loc, left, right)) {
            freeTrees(2, left, right);
            return false;
        }
        return true;
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
    if (tree->isLeaf) {
        return copyTree(result, tree);
    }

    Split_T split = tree->split;
    if (axis != split->axis) {
        /* Prune both subtrees. */
        Tree_T left, right;
        if (!treePruneRightSafe(&left, split->left, axis, loc)) {
            return false;
        }
        if (!treePruneRightSafe(&right, split->right, axis, loc)) {
            freeTree(left);
            return false;
        }
        if (!makeSplit(result, tree->dim, split->axis,
                         split->loc, left, right)) {
            freeTrees(2, left, right);
            return false;
        }
        return true;
    }

    if (loc <= split->loc) {
        /* Prune left subtree, copy right subtree. */
        Tree_T left, right;
        if (!treePruneRightSafe(&left, split->left, axis, loc)) {
            return false;
        }
        if (!copyTreeSafe(&right, split->right)) {
            freeTree(left);
            return false;
        }
        if (!makeSplit(result, tree->dim, split->axis,
                       split->loc, left, right)) {
            freeTrees(2, left, right);
        }
        return true;
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

void treeAddConstAux(Tree_T tree, double c) {
    if (tree->isLeaf) {
        tree->val += c;
        return;
    }

    tree->split->min += c;
    tree->split->max += c;
    treeAddConstAux(tree->split->left, c);
    treeAddConstAux(tree->split->right, c);
}

/* Creates a copy of a tree where a constant has been added to all leaves. */
bool treeAddConst(Tree_T * result, Tree_T tree, double c) {
    assert(result != NULL);
    assert(validTree(tree));

    if (!copyTree(result, tree)) {
        return false;
    }

    treeAddConstAux(*result, c);
    return true;
}

bool treeMergeSafe(Tree_T * result, Tree_T tree1, Tree_T tree2) {
    *result = NULL;
    uint32_t dim = tree1->dim;

    /* One or both trees are leaves. */
    if (tree1->isLeaf) {
        return treeAddConst(result, tree2, tree1->val);
    }
    if (tree2->isLeaf) {
        return treeAddConst(result, tree1, tree2->val);
    }

    Split_T split1 = tree1->split;
    Split_T split2 = tree2->split;
    /* Different splitting axes. */
    if (split1->axis != split2->axis) {
        /* Merge left and right subtrees from tree1 with tree2. */
        Tree_T mergeL, mergeR, pruneL, pruneR;
        if (!treePruneLeftSafe(&pruneL, tree2, split1->axis, split1->loc)) {
            return false;
        }
        if (!treeMergeSafe(&mergeL, split1->left, pruneL)) {
            freeTree(pruneL);
            return false;
        }
        freeTree(pruneL);

        if (!treePruneRightSafe(&pruneR, tree2, split1->axis, split1->loc)) {
            freeTree(mergeL);
            return false;
        }
        if (!treeMergeSafe(&mergeR, split1->right, pruneR)) {
            freeTrees(2, mergeL, pruneR);
            return false;
        }
        freeTree(pruneR);


        if (!makeSplit(result, dim, split1->axis, split1->loc, mergeL, mergeR)) {
            freeTrees(2, mergeL, mergeR);
            return false;
        }
        return true;
    }

    /* Tree 1 and Tree 2 split on same axis. */
    uint32_t axis = split1->axis;
    if (split1->loc < split2->loc) {
        /* Make 3 subtrees (left, center, right). */
        Tree_T center1, right1, left2, center2;
        Tree_T mergeL, mergeC, mergeR;

        /* Left subtree. */
        if (!treePruneLeft(&left2, split2->left, axis, split1->loc)) {
            return false;
        }
        if (!treeMergeSafe(&mergeL, split1->left, left2)) {
            freeTree(left2);
            return false;
        }
        freeTree(left2);

        /* Center subtree. */
        if (!treePruneLeft(&center1, split1->right, axis, split2->loc)) {
            freeTree(mergeL);
            return false;
        }
        if (!treePruneRight(&center2, split2->left, axis, split1->loc)) {
            freeTrees(2, mergeL, center1);
            return false;
        }
        if (!treeMergeSafe(&mergeC, center1, center2)) {
            freeTrees(3, mergeL, center1, center2);
            return false;
        }
        freeTrees(2, center1, center2);

        /* Right subtree. */
        if (!treePruneRight(&right1, split1->right, axis, split2->loc)) {
            freeTrees(2, mergeL, mergeC);
            return false;
        }
        if (!treeMergeSafe(&mergeR, right1, split2->right)) {
            freeTrees(3, mergeL, mergeC, right1);
            return false;
        }
        freeTree(right1);

        /* Connect 3 subtrees. Balance based on size. */
        if (treeSize(mergeL) > treeSize(mergeR)) {
            Tree_T mergeCR;
            assert(validTree(mergeL));
            assert(validTree(mergeC));
            assert(validTree(mergeR));
            if (!makeSplit(&mergeCR, dim, axis, split2->loc, mergeC, mergeR)) {
                freeTrees(3, mergeL, mergeC, mergeR);
                return false;
            }
            if (!makeSplit(result, dim, axis, split1->loc, mergeL, mergeCR)) {
                freeTrees(2, mergeL, mergeCR);
                return false;
            }
        } else {
            Tree_T mergeLC;
            if (!makeSplit(&mergeLC, dim, axis, split1->loc, mergeL, mergeC)) {
                freeTrees(3, mergeL, mergeC, mergeR);
                return false;
            }
            if (!makeSplit(result, dim, axis, split2->loc, mergeLC, mergeR)) {
                freeTrees(2, mergeLC, mergeR);
                return false;
            }
        }
    } else if (split2->loc < split1->loc) {
        /* Make 3 subtrees (left, center, right). */
        Tree_T left1, center1, center2, right2;
        Tree_T mergeL, mergeC, mergeR;

        /* Left subtree. */
        if (!treePruneLeft(&left1, split1->left, axis, split2->loc)) {
            return false;
        }
        if (!treeMergeSafe(&mergeL, left1, split2->left)) {
            freeTree(left1);
            return false;
        }
        freeTree(left1);

        /* Center subtree. */
        if (!treePruneRight(&center1, split1->left, axis, split2->loc)) {
            freeTree(mergeL);
            return false;
        }
        if (!treePruneLeft(&center2, split2->right, axis, split1->loc)) {
            freeTrees(2, mergeL, center1);
            return false;
        }
        if (!treeMergeSafe(&mergeC, center1, center2)) {
            freeTrees(3, mergeL, center1, center2);
            return false;
        }
        freeTrees(2, center1, center2);

        /* Right subtree. */
        if (!treePruneRight(&right2, split2->right, axis, split1->loc)) {
            freeTrees(2, mergeL, mergeC);
            return false;
        }
        if (!treeMergeSafe(&mergeR, split1->right, right2)) {
            freeTrees(3, mergeL, mergeC, right2);
            return false;
        }

        /* Connect 3 subtrees. Balance based on size. */
        if (treeSize(mergeL) > treeSize(mergeR)) {
            Tree_T mergeCR;
            if (!makeSplit(&mergeCR, dim, axis, split1->loc, mergeC, mergeR)) {
                freeTrees(3, mergeL, mergeC, mergeR);
                return false;
            }
            if (!makeSplit(result, dim, axis, split2->loc, mergeL, mergeCR)) {
                freeTrees(2, mergeL, mergeCR);
                return false;
            }
        } else {
            Tree_T mergeLC;
            if (!makeSplit(&mergeLC, dim, axis, split2->loc, mergeL, mergeC)) {
                freeTrees(3, mergeL, mergeC, mergeR);
                return false;
            }
            if (!makeSplit(result, dim, axis, split1->loc, mergeLC, mergeR)) {
                freeTrees(2, mergeLC, mergeR);
                return false;
            }
        }        
    } else {
        /* Merge left subtrees together znd right subtrees together.*/
        Tree_T left, right;
        if (!treeMergeSafe(&left, split1->left, split2->left)) {
            return false;
        }
        if (!treeMergeSafe(&right, split1->right, split2->right)) {
            freeTree(left);
            return false;
        }
        if (!makeSplit(result, dim, split1->axis, split1->loc, left, right)) {
            freeTrees(2, left, right);
            return false;
        }
    }
    return true;
}

bool treeMerge(Tree_T * result, Tree_T tree1, Tree_T tree2) {
    /* Only validate arguments once. */
    assert(result != NULL);
    assert(validTree(tree1));
    assert(validTree(tree2));
    assert(tree1->dim == tree2->dim);

    return treeMergeSafe(result, tree1, tree2);
}
