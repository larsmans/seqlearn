"""Utils."""

cimport cython
cimport numpy as np
import numpy as np

np.import_array()

# TODO handle second-order transitions (trigrams)
@cython.boundscheck(False)
@cython.wraparound(False)
def count_trans(np.ndarray[ndim=1, dtype=np.int32_t] y, int n_classes):
    """Count transitions in a target vector.

    Parameters
    ----------
    y : array, shape = n_samples
    n_classes : int
        Number of distinct labels.
    """
    cdef int i
    cdef np.ndarray[ndim=2, dtype=np.int32_t, mode='c'] trans
    trans = np.zeros((n_classes, n_classes), dtype=np.int32)

    for i in range(len(y) - 1):
        trans[y[i], y[i + 1]] += 1
    return trans

