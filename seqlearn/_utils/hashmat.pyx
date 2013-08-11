# Copyright 2013 Lars Buitinck

# Sparse matrices implemented as hash tables, to be used for storing
# parameters.

from libc.errno cimport errno, ENOMEM

cdef extern from "hashmatrix_c.h":
    ctypedef struct Matrix:
        size_t ncols, nrows
    int add(Matrix *, Matrix *, double)
    void add_to_dense(double *, Matrix *)
    size_t count(const Matrix *)
    void destroy(Matrix *)
    double get(Matrix *, size_t, size_t)
    void initialize(Matrix *, size_t, size_t)
    void put(Matrix *, size_t, size_t, double)

cimport numpy as np
import numpy as np

np.import_array()


cdef class HashMatrix:
    """Sparse matrix of doubles implemented as a hash table."""

    cdef Matrix mat

    def __cinit__(self):
        initialize(&self.mat, 0, 0)

    def __init__(self, shape):
        cdef size_t n, m

        n, m = shape[0], shape[1]
        initialize(&self.mat, n, m)

    def __dealloc__(self):
        destroy(&self.mat)

    cdef void _add(self, HashMatrix other, double factor) except *:
        if add(&self.mat, &other.mat, factor) != 0:
            if errno == ENOMEM:
                raise MemoryError()
            else:
                raise SystemError()

    def __getitem__(self, idx):
        i, j = idx
        return get(&self.mat, i, j)

    def getnnz(self):
        return count(&self.mat)

    def __iadd__(self, HashMatrix other):
        self._add(other, 1)
        return self

    def __isub__(self, HashMatrix other):
        self._add(other, -1)
        return self

    @property
    def nnz(self):
        return self.getnnz()

    def __setitem__(self, idx, x):
        i, j = idx
        put(&self.mat, i, j, x)

    @property
    def shape(self):
        return self.mat.nrows, self.mat.ncols

    def toarray(self):
        cdef np.ndarray[ndim=1, dtype=double, mode="c"] dense
        dense = np.zeros(self.mat.nrows * self.mat.ncols)
        add_to_dense(<double *>dense.data, &self.mat)
        return dense.reshape(-1, self.mat.ncols)

    @property
    def T(self):
        return TransposedHashMatrix(self)

    def transpose(self):
        return self.T


cdef class TransposedHashMatrix:
    cdef HashMatrix base

    def __cinit__(self):
        pass

    def __init__(self, HashMatrix base):
        self.base = base

    def getnnz(self):
        return self.base.getnnz()

    @property
    def nnz(self):
        return self.base.getnnz()

    def shape(self):
        n, m = self.base.shape
        return m, n

    def toarray(self):
        return self.base.toarray().T

    @property
    def T(self):
        return self.base

    def transpose(self):
        return self.base


def dot_csr_hash(X, HashMatrix Y):
    """Dot product of X and Y."""

    cdef:
        np.ndarray[ndim=1, dtype=double, mode="c"] data = X.data
        np.ndarray[ndim=1, dtype=int, mode="c"] indices = X.indices
        np.ndarray[ndim=1, dtype=int, mode="c"] indptr = X.indptr

        cdef double sum_, x, y
        np.npy_intp i, j, jj, k

        Matrix *B = &(Y.mat)

        np.ndarray[ndim=2, dtype=double, mode="c"] R

    R = np.zeros((indptr.shape[0] - 1, B.ncols), order="C")

    for i in range(indptr.shape[0] - 1):
        for k in range(B.ncols):
            sum_ = 0.

            for jj in range(indptr[i], indptr[i + 1]):
                j = indices[jj]
                sum_ += data[jj] * get(B, j, k)

            R[i, k] = sum_

    return R
