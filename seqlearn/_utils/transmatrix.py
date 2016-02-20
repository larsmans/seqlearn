# Copyright 2013 Lars Buitinck / University of Amsterdam
# encoding: utf-8

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.externals import six


def make_trans_matrix(y, n_classes, dtype=np.float64):
    """Make a sparse transition matrix for y.

    Takes a label sequence y and returns an indicator matrix with n_classes²
    columns of the label transitions in y: M[i, j, k] means y[i-1] == j and
    y[i] == k. The first row will be empty.
    """
    indices = np.empty(len(y), dtype=np.int32)

    for i in six.moves.xrange(len(y) - 1):
        indices[i] = y[i] * i + y[i + 1]

    indptr = np.arange(len(y) + 1)
    indptr[-1] = indptr[-2]

    return csr_matrix((np.ones(len(y), dtype=dtype), indices, indptr),
                      shape=(len(y), n_classes ** 2))


def make_trans_mask(trans_constraints, classes):
    """ Given a list of tuples that match elements in the list classes

    Parameters
    ----------
    trans_constraints : list
        A list of tuples of length two.  The first element is the prev_state,
        the latter element is the current_state.  The existance of a constraint
        pair (prev_state, current_state) significantly lowers the transition
        probability between elements

    classes : list
        The list of classes

    """
    n_classes = len(classes)
    classdict = {c:i for i,c in enumerate(classes)}

    trans_mask = np.zeros((n_classes, n_classes), dtype=int)

    for src, dest in trans_constraints:
        r = classdict.get(src,-1)
        c = classdict.get(dest,-1)

        # Check if valid constraint
        if r > -1 and c > -1:
            trans_mask[r,c] = 1

    return trans_mask