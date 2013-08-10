# encoding: utf-8

import numpy as np
from scipy.sparse import csr_matrix


def make_trans_matrix(y, n_classes, dtype=np.float64):
    """Make a sparse transition matrix for y.

    Takes a label sequence y and returns an indicator matrix with
    n_classes × (n_classes + 1) columns of the label transitions in y.
    """
    indices = np.empty(len(y) + 1, dtype=np.int32)

    n_states = n_classes + 1
    START = n_states - 1
    #START = n_states - 2
    #END = n_states - 1

    def trans_idx(i, j):
        return n_states * i + j

    indices[0] = trans_idx(START, y[0])
    for i in xrange(1, len(y)):
        indices[i] = trans_idx(y[i - 1], y[i])
    #indices[-1] = trans_idx(y[i - 1], END)

    return csr_matrix((np.ones(len(y) + 1, dtype=dtype),
                       indices,
                       np.arange(len(y) + 1)),
                      shape=(len(y), n_states ** 2))


def unroll_trans_matrix(Y):
    n_samples = Y.shape[0]
    n_states = int(np.sqrt(Y.shape[1]))
    n_classes = n_states - 1

    y = np.empty(n_samples, dtype=np.intp)
    y[0] = Y.data[0] / n_states
    for i in xrange(1, n_samples):
        y[i] = Y.indices[i] - y[i - 1] * n_states

    return y
