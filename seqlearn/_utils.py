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
