import numpy as np

# XXX These are private helper functions from scikit-learn. We should copy
# the code over instead of importing them.
from sklearn.utils import atleast2d_or_csr, check_random_state
from sklearn.utils.extmath import logsumexp, safe_sparse_dot


# TODO handle second-order transitions (trigrams)
def count_trans(y, n_classes):
    """Count transitions in a target vector.

    Parameters
    ----------
    y : array, shape = n_samples
    n_classes : int
        Number of distinct labels.
    """
    trans = np.zeros((n_classes, n_classes), dtype=np.int32)
    for i in xrange(len(y) - 1):
        trans[y[i], y[i + 1]] += 1
    return trans


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
