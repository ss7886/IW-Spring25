#include <stdbool.h>
#include <stdint.h>

/* Only include __declspec(dllexport) if compiling on windows. */
#ifdef _MSC_VER
    #define EXPORT_SYMBOL __declspec(dllexport)
#else
    #define EXPORT_SYMBOL
#endif

#ifndef TREE_INCLUDED
#define TREE_INCLUDED

EXPORT_SYMBOL typedef struct tree * Tree_T;
EXPORT_SYMBOL typedef struct split * Split_T;

EXPORT_SYMBOL struct tree {
    bool isLeaf;
    uint32_t dim;
    double val;
    Split_T split;
};

EXPORT_SYMBOL struct split {
    uint32_t axis;
    double loc;
    Tree_T left;
    Tree_T right;
    double min;
    double max;
    uint32_t depth;
    uint32_t size;
};

EXPORT_SYMBOL double treeMin(Tree_T tree);
EXPORT_SYMBOL double treeMax(Tree_T tree);
EXPORT_SYMBOL uint32_t treeDepth(Tree_T tree);
EXPORT_SYMBOL uint32_t treeSize(Tree_T tree);

EXPORT_SYMBOL bool validTree(Tree_T tree);
EXPORT_SYMBOL bool validLeaf(Tree_T tree);
EXPORT_SYMBOL bool validSplit(Tree_T tree);

EXPORT_SYMBOL bool makeLeaf(Tree_T * result, uint32_t dim, double val);
EXPORT_SYMBOL bool makeSplit(Tree_T * result, uint32_t dim, uint32_t axis,
               double loc, Tree_T left, Tree_T right);
EXPORT_SYMBOL bool copyTree(Tree_T * result, Tree_T tree);
EXPORT_SYMBOL void freeTree(Tree_T tree);
EXPORT_SYMBOL void freeTrees(int count, ...);

EXPORT_SYMBOL double treeEval(Tree_T tree, double x[]);

EXPORT_SYMBOL bool treePruneLeft(Tree_T * result, Tree_T tree, uint32_t axis,
                                 double loc);
EXPORT_SYMBOL bool treePruneRight(Tree_T * result, Tree_T tree, uint32_t axis,
                                  double loc);
EXPORT_SYMBOL bool treeMerge(Tree_T * result, Tree_T tree1, Tree_T tree2);

#endif
