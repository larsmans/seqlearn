/* Copyright 2013 Lars Buitinck */

#include <stdlib.h>
#include "uthash.h"

/*
 * Sparse matrix of doubles implemented using as a hash table.
 * This is similar to scipy.sparse's dok_matrix type, but much more
 * efficient and with additional operations.
 */
typedef struct {
    // coordinates, rolled into one size_t to be hashable
    size_t ij;
    double v;
    UT_hash_handle hh;
} Entry;

typedef struct {
    size_t ncols, nrows;
    Entry *table;           // hash table of entries
} Matrix;

static void initialize(Matrix *m, size_t nr, size_t nc)
{
    m->ncols = nc;
    m->nrows = nr;
    m->table = NULL;
}

static int put_ij(Matrix *A, size_t ij, double v)
{
    Entry *p;

    HASH_FIND(hh, A->table, &ij, sizeof(ij), p);
    if (p == NULL) {
        p = malloc(sizeof(Entry));
        if (p == NULL)
            return -1;
        p->ij = ij;
        HASH_ADD(hh, A->table, ij, sizeof(ij), p);
    }
    p->v = v;

    return 0;
}

// A += factor * B
static int add(Matrix *A, Matrix *B, double factor)
{
    Entry *b;

    for (b = B->table; b; b = b->hh.next) {
        double v = factor * b->v;
        Entry *a;
        HASH_FIND(hh, A->table, &(b->ij), sizeof(b->ij), a);

        if (a)
            a->v += v;
        else if (put_ij(A, b->ij, v) != 0)
            return -1;
    }
    return 0;
}

// Number of stored elements.
static size_t count(Matrix const *A)
{
    return HASH_COUNT(A->table);
}

static void destroy(Matrix *A)
{
    Entry *a, *tmp;
    HASH_ITER(hh, A->table, a, tmp) {
        HASH_DEL(A->table, a);
        free(a);
    }
}

static double get(Matrix *A, size_t i, size_t j)
{
    Entry *a;
    size_t ij = i * A->ncols + j;

    HASH_FIND(hh, A->table, &ij, sizeof(ij), a);
    return a == NULL? 0. : a->v;
}

// A[i, j] = v
static int put(Matrix *A, size_t i, size_t j, double v)
{
    return put_ij(A, i * A->ncols + j, v);
}

// Add B to unraveled, C-contiguous array A.
static void add_to_dense(double *A, Matrix *B)
{
    Entry *b;
    for (b = B->table; b; b = b->hh.next) {
        A[b->ij] += b->v;
    }
}
