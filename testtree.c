#include <assert.h>
#include <stdio.h>
#include "tree.h"

int main(void) {
    /* Make Tree. */
    Tree_T leafA, leafB, leafC, leafD, leafE, leafF;
    assert(makeLeaf(&leafA, 3, 0.1));
    assert(makeLeaf(&leafB, 3, 0.2));
    assert(makeLeaf(&leafC, 3, 0.3));
    assert(makeLeaf(&leafD, 3, 0.4));
    assert(makeLeaf(&leafE, 3, 0.5));
    assert(makeLeaf(&leafF, 3, 0.6));
    Tree_T tree, treeL, treeR, treeRL, treeRR;
    assert(makeSplit(&treeRL, 3, 0, 0.0, leafC, leafD));
    assert(makeSplit(&treeRR, 3, 2, 1.0, leafE, leafF));
    assert(makeSplit(&treeL, 3, 2, -1.0, leafA, leafB));
    assert(makeSplit(&treeR, 3, 0, 1.0, treeRL, treeRR));
    assert(makeSplit(&tree, 3, 1, 0.0, treeL, treeR));

    /* Test values. */
    assert(treeMin(tree) == 0.1);
    assert(treeMax(tree) == 0.6);    
    assert(treeDepth(tree) == 3);
    assert(treeSize(tree) == 6);

    assert(treeMin(treeL) == 0.1);
    assert(treeMax(treeL) == 0.2);    
    assert(treeDepth(treeL) == 1);
    assert(treeSize(treeL) == 2);

    assert(treeMin(treeR) == 0.3);
    assert(treeMax(treeR) == 0.6);    
    assert(treeDepth(treeR) == 2);
    assert(treeSize(treeR) == 4);

    assert(treeMin(treeRL) == 0.3);
    assert(treeMax(treeRL) == 0.4);    
    assert(treeDepth(treeRL) == 1);
    assert(treeSize(treeRL) == 2);

    assert(treeMin(treeRR) == 0.5);
    assert(treeMax(treeRR) == 0.6);    
    assert(treeDepth(treeRR) == 1);
    assert(treeSize(treeRR) == 2);

    /* Test eval. */    
    double x1[] = {0.0, 0.0, -1.0};
    double x2[] = {1.0, -1.0, 0.0};
    double x3[] = {0.0, 1.0, 1.0};
    double x4[] = {1.0, 1.0, 0.0};
    double x5[] = {2.0, 1.0, 1.0};
    double x6[] = {2.0, 1.0, 2.0};

    assert(treeEval(tree, x1) == 0.1);
    assert(treeEval(tree, x2) == 0.2);
    assert(treeEval(tree, x3) == 0.3);
    assert(treeEval(tree, x4) == 0.4);
    assert(treeEval(tree, x5) == 0.5);
    assert(treeEval(tree, x6) == 0.6);

    /* Test copy. */
    Tree_T copy;
    assert(copyTree(&copy, tree));

    assert(treeEval(copy, x1) == 0.1);
    assert(treeEval(copy, x2) == 0.2);
    assert(treeEval(copy, x3) == 0.3);
    assert(treeEval(copy, x4) == 0.4);
    assert(treeEval(copy, x5) == 0.5);
    assert(treeEval(copy, x6) == 0.6);

    /* Test prune left. */
    Tree_T pruneLeft;
    assert(treePruneLeft(&pruneLeft, tree, 0, 0.5));
    assert(treeDepth(pruneLeft) == 2);
    assert(treeSize(pruneLeft) == 4);

    assert(treeEval(pruneLeft, x1) == 0.1);
    assert(treeEval(pruneLeft, x2) == 0.2);
    assert(treeEval(pruneLeft, x3) == 0.3);
    assert(treeEval(pruneLeft, x4) == 0.4);
    assert(treeEval(pruneLeft, x5) == 0.4);
    assert(treeEval(pruneLeft, x6) == 0.4);

    /* Test prune right. */
    Tree_T pruneRight;
    assert(treePruneRight(&pruneRight, tree, 0, 0.5));
    assert(treeDepth(pruneRight) == 3);
    assert(treeSize(pruneRight) == 5);

    assert(treeEval(pruneRight, x1) == 0.1);
    assert(treeEval(pruneRight, x2) == 0.2);
    assert(treeEval(pruneRight, x3) == 0.4);
    assert(treeEval(pruneRight, x4) == 0.4);
    assert(treeEval(pruneRight, x5) == 0.5);
    assert(treeEval(pruneRight, x6) == 0.6);

    /* Test that copies are still ok after freeing tree. */
    freeTree(tree);
    assert(treeEval(copy, x1) == 0.1);
    assert(treeEval(pruneLeft, x1) == 0.1);
    assert(treeEval(pruneRight, x1) == 0.1);

    /* Free memory. */
    freeTree(copy);
    freeTree(pruneLeft);
    freeTree(pruneRight);

    printf("Tests passed!\n");
}
