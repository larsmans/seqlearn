# Copyright 2013 Lars Buitinck

import numpy as np
from scipy.sparse import issparse

# XXX These are private helper functions from scikit-learn. We should copy
# the code over instead of importing them.
from sklearn.utils import atleast2d_or_csr, check_random_state
from sklearn.utils.extmath import logsumexp

from .ctrans import count_trans
from .dokmatrix import DOKMatrix, dot_csr_dok, dot_dense_dok
from .safeadd import safe_add


def safe_sparse_dot(A, B, dense_output=False):
    """Dot product of A and B, which may be dense or sparse.

    Exception: A may not be a DOKMatrix.
    """
    if isinstance(B, DOKMatrix):
        if issparse(A):
            return dot_csr_dok(A.tocsr(), B)
        return dot_dense_dok(A, B)
    if issparse(A) or issparse(B):
        C = A * B
        if dense_output and issparse(C):
            C = C.toarray()
        return C
    return np.dot(A, B)


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
