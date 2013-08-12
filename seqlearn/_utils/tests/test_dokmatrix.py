from __future__ import division

from nose.tools import assert_equal
from numpy.testing import assert_array_equal

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix

from seqlearn._utils.dokmatrix import dot_csr_dok, DOKMatrix


def test_dokmatrix():
    l, m, n = 5, 7, 13

    Y = DOKMatrix((m, n))
    Yd = np.zeros((m, n), dtype=np.float)

    nnz = 0
    for i in xrange(l):
        for j in xrange(m):
            Yd[i, j] = i * j / 3.4
            Y[i, j] = Yd[i, j]
            nnz += 1
            assert_equal(nnz, Y.nnz)

    assert_equal(Yd.shape, Y.shape)
    assert_equal(Yd[1, 2], Y[1, 2])
    assert_array_equal(Yd, Y.toarray())

    X = csr_matrix(np.random.randn(l * m).reshape(l, m), dtype=np.float)

    assert_array_equal(X * Yd, dot_csr_dok(X, Y))


def test_add_subtract():
    X = DOKMatrix((50, 1800))
    Y = DOKMatrix((50, 1800))

    Y[42, 1707] = np.e
    X += Y
    assert_array_equal(X.toarray(), Y.toarray())

    Y[37, 1789] = np.pi
    X -= Y
    assert_equal(-np.pi, X[37, 1789])
    assert_equal(2, Y.getnnz())


def test_add_csc_div():
    X = DOKMatrix((3, 4))
    rng = np.arange(12, dtype=np.float64).reshape(4, 3)
    X.T.add(csc_matrix(rng), 1)

    assert_array_equal(X.toarray().T, rng)

    X /= 2.
    assert_array_equal(X.toarray().T, rng / 2.)
