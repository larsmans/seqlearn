# distutils: language = c++

# Copyright 2013 Lars Buitinck

# Sparse matrices implemented as binary search trees, to be used for storing
# parameters.

cimport cython
from libc.errno cimport errno, ENOMEM

cdef extern from "bstmatrix.hpp" namespace "seqlearn":
    cdef cppclass BSTMatrix:
        double add(BSTMatrix *, double)
        double add(size_t, size_t, double)
        double add_to_dense(double *, size_t)
        double get(size_t i, size_t j)
        void mul(double)
        void put(size_t, size_t, double) except +
        size_t size()

cimport numpy as np
import numpy as np
import scipy.sparse as sp

np.import_array()


cdef class DOKMatrix:
    """Sparse matrix of doubles implemented as a binary search tree.

    Like scipy.sparse.dok_matrix, but much more efficient and with some
    useful operations added.
    """

    cdef BSTMatrix *mat
    cdef np.npy_intp ncols, nrows

    def __cinit__(self):
        self.mat = new BSTMatrix()

    def __init__(self, shape):
        cdef size_t n, m

        n, m = shape[0], shape[1]
        self.nrows = n
        self.ncols = m

    def __dealloc__(self):
        del self.mat

    def __getitem__(self, idx):
        i, j = idx
        return self.mat.get(i, j)

    def getnnz(self):
        return self.mat.size()

    def __iadd__(self, DOKMatrix other):
        self.mat.add(other.mat, 1)
        return self

    def __itruediv__(self, v):
        self.mat.mul(1. / v)
        return self

    def __isub__(self, DOKMatrix other):
        self.mat.add(other.mat, -1)
        return self

    @property
    def nnz(self):
        return self.getnnz()

    def __setitem__(self, idx, x):
        i, j = idx
        self.mat.put(i, j, x)

    @property
    def shape(self):
        return self.nrows, self.ncols

    def toarray(self):
        cdef np.ndarray[ndim=1, dtype=double, mode="c"] dense
        dense = np.zeros(self.nrows * self.ncols)
        self.mat.add_to_dense(<double *>dense.data, self.ncols)
        return dense.reshape(-1, self.ncols)

    @property
    def T(self):
        return TransposedDOKMatrix(self)

    def transpose(self):
        return self.T


cdef class TransposedDOKMatrix:
    cdef DOKMatrix base

    def __cinit__(self):
        pass

    def __init__(self, DOKMatrix base):
        self.base = base

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add(self, X, double factor):
        """Add CSC matrix X * factor (scalar) to self."""

        if not sp.isspmatrix_csc(X):
            raise TypeError("Addition only implemented for CSC matrices.")

        cdef:
            BSTMatrix *mat = self.base.mat
            np.ndarray[ndim=1, dtype=np.float64_t, mode="c"] data = X.data
            np.ndarray[ndim=1, dtype=int, mode="c"] indices = X.indices, \
                                                    indptr = X.indptr
            np.npy_intp i, j, jj

        for i in range(indptr.shape[0] - 1):
            for jj in range(indptr[i], indptr[i + 1]):
                j = indices[jj]
                mat.add(i, j, factor * data[jj])

    def getnnz(self):
        return self.base.getnnz()

    def __iadd__(self, TransposedDOKMatrix other):
        self.base += other.base
        return self

    def __itruediv__(self, v):
        self.base.mat.mul(1. / v)
        return self

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


@cython.boundscheck(False)
@cython.wraparound(False)
def dot_csr_dok(X, DOKMatrix Y):
    """Dot product of X and Y."""

    cdef:
        np.ndarray[ndim=1, dtype=double, mode="c"] data = X.data
        np.ndarray[ndim=1, dtype=int, mode="c"] indices = X.indices
        np.ndarray[ndim=1, dtype=int, mode="c"] indptr = X.indptr

        double s, x, y
        np.npy_intp i, j, jj, k

        BSTMatrix *B = Y.mat

        np.ndarray[ndim=2, dtype=double, mode="c"] R

    if X.shape[1] != Y.nrows:
        raise ValueError("Shape mismatch: %s, %s" % (X.shape, Y.shape))

    R = np.zeros((indptr.shape[0] - 1, Y.ncols), order="C")

    for i in range(indptr.shape[0] - 1):
        for k in range(Y.ncols):
            s = 0.

            for jj in range(indptr[i], indptr[i + 1]):
                j = indices[jj]
                s += data[jj] * B.get(j, k)

            R[i, k] = s

    return R


@cython.boundscheck(False)
@cython.wraparound(False)
def dot_dense_dok(np.ndarray[ndim=2, dtype=np.float64_t] X, DOKMatrix Y):
    cdef:
        double s
        np.npy_intp i, j, k
        np.ndarray[ndim=2, dtype=double, mode="c"] R

    R = np.zeros((X.shape[0], Y.ncols), order="C")

    # XXX this can be done potentially faster if we switch to std::map.
    # It's not the primary use case, though.
    for i in xrange(X.shape[0]):
        for k in range(Y.ncols):
            s = 0.
            for j in range(X.shape[1]):
                s += X[i, j] * Y.mat.get(j, k)

    return R
