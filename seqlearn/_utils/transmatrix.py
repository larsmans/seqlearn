# Copyright 2013 Lars Buitinck / University of Amsterdam
# encoding: utf-8

import numpy as np
from scipy.sparse import csr_matrix


def make_trans_matrix(y, n_classes, dtype=np.float64):
    """Make a sparse transition matrix for y.

    Takes a label sequence y and returns an indicator matrix with n_classes²
    columns of the label transitions in y: M[i, j, k] means y[i-1] == j and
    y[i] == k. The first row will be empty.
    """
    indices = np.empty(len(y), dtype=np.int32)

    for i in xrange(len(y) - 1):
        indices[i] = y[i] * i + y[i + 1]

    indptr = np.arange(len(y) + 1)
    indptr[-1] = indptr[-2]

    return csr_matrix((np.ones(len(y), dtype=dtype), indices, indptr),
                      shape=(len(y), n_classes ** 2))
