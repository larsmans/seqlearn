# Copyright 2013-2014 Lars Buitinck / University of Amsterdam
#
# Parts taken from scikit-learn, written by
#          Olivier Grisel
#          Gael Varoquaux
#          Andreas Mueller
#          Lars Buitinck
#          Alexandre Gramfort

import numpy as np
import scipy.sparse as sp

# XXX These are private helper functions from scikit-learn. We should copy
# the code over instead of importing them.
from sklearn.utils import check_random_state
from sklearn.utils.extmath import safe_sparse_dot

from .ctrans import count_trans
from .safeadd import safe_add
from .transmatrix import make_trans_matrix


def _assert_all_finite(X):
    X = np.asanyarray(X)
    # First try an O(n) time, O(1) space solution for the common case that
    # there everything is finite; fall back to O(n) space np.isfinite to
    # prevent false positives from overflow in sum method.
    if (X.dtype.char in np.typecodes['AllFloat'] and not np.isfinite(X.sum())
            and not np.isfinite(X).all()):
        raise ValueError("Input contains NaN, infinity"
                         " or a value too large for %r." % X.dtype)


def assert_all_finite(X):
    """Throw a ValueError if X contains NaN or infinity.

    Input MUST be an np.ndarray instance or a scipy.sparse matrix."""

    _assert_all_finite(X.data if sp.issparse(X) else X)


def array2d(X, dtype=None, order=None, copy=False):
    """Returns at least 2-d array with data from X"""
    if sp.issparse(X):
        raise TypeError('A sparse matrix was passed, but dense data '
                        'is required. Use X.toarray() to convert to dense.')
    X_2d = np.asarray(np.atleast_2d(X), dtype=dtype, order=order)
    _assert_all_finite(X_2d)
    if X is X_2d and copy:
        X_2d = safe_copy(X_2d)
    return X_2d


def _atleast2d_or_sparse(X, dtype, order, copy, sparse_class, convmethod,
                         check_same_type):
    if sp.issparse(X):
        if check_same_type(X) and X.dtype == dtype:
            X = getattr(X, convmethod)(copy=copy)
        elif dtype is None or X.dtype == dtype:
            X = getattr(X, convmethod)()
        else:
            X = sparse_class(X, dtype=dtype)
        _assert_all_finite(X.data)
        X.data = np.array(X.data, copy=False, order=order)
    else:
        X = array2d(X, dtype=dtype, order=order, copy=copy)
    return X


def atleast2d_or_csr(X, dtype=None, order=None, copy=False):
    """Like numpy.atleast_2d, but converts sparse matrices to CSR format

    Also, converts np.matrix to np.ndarray.
    """
    return _atleast2d_or_sparse(X, dtype, order, copy, sp.csr_matrix,
                                "tocsr", sp.isspmatrix_csr)


def validate_lengths(n_samples, lengths):
    """Validate lengths array against n_samples.

    Parameters
    ----------
    n_samples : integer
        Total number of samples.

    lengths : array-like of integers, shape (n_sequences,), optional
        Lengths of individual sequences in the input.

    Returns
    -------
    start : array of integers, shape (n_sequences,)
        Start indices of sequences.

    end : array of integers, shape (n_sequences,)
        One-past-the-end indices of sequences.
    """
    if lengths is None:
        lengths = [n_samples]
    lengths = np.asarray(lengths, dtype=np.int32)
    if lengths.sum() > n_samples:
        msg = "More than {0:d} samples in lengths array {1!s}"
        raise ValueError(msg.format(n_samples, lengths))

    end = np.cumsum(lengths)
    start = end - lengths

    return start, end
