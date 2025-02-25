#include <stdbool.h>
#include <stdint.h>

#ifndef TREE_INCLUDED
#define TREE_INCLUDED

typedef struct tree * Tree_T;
typedef struct split * Split_T;

struct tree {
    bool isLeaf;
    uint32_t dim;
    double val;
    Split_T split;
};

struct split {
    uint32_t axis;
    double loc;
    Tree_T left;
    Tree_T right;
    double min;
    double max;
    uint32_t depth;
    uint32_t size;
};

double treeMin(const Tree_T tree);
double treeMax(const Tree_T tree);
uint32_t treeDepth(const Tree_T tree);
uint32_t treeSize(const Tree_T tree);

bool validTree(const Tree_T tree);
bool validLeaf(const Tree_T tree);
bool validSplit(const Tree_T tree);

bool makeLeaf(Tree_T * result, uint32_t dim, double val);
bool makeSplit(Tree_T * result, uint32_t dim, uint32_t axis,
               double loc, const Tree_T left, const Tree_T right);
bool copyTree(Tree_T * result, const Tree_T tree);
void freeTree(Tree_T tree);
void freeTrees(int count, ...);

double treeEval(const Tree_T tree, const double x[]);

bool treePruneLeft(Tree_T * result, const Tree_T tree, uint32_t axis,
                                 double loc);
Tree_T treePruneLeftInPlace(Tree_T tree, uint32_t axis, double loc);
bool treePruneRight(Tree_T * result, const Tree_T tree, uint32_t axis,
                                  double loc);
Tree_T treePruneRightInPlace(Tree_T tree, uint32_t axis, double loc);
bool treeMerge(Tree_T * result, const Tree_T tree1, const Tree_T tree2);

void featureImportance(double importances[], const Tree_T tree);

#endif
