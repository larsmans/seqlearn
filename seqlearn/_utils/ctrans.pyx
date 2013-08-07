# Copyright Lars Buitinck, Mikhail Korobov 2013.

cimport cython
cimport numpy as np
import numpy as np

np.import_array()

# TODO handle second-order transitions (trigrams)
@cython.boundscheck(False)
@cython.wraparound(False)
def count_trans(np.ndarray[ndim=1, dtype=np.npy_intp] y, n_classes):
    """Count transitions in a target vector.

    Parameters
    ----------
    y : array of integers, shape = n_samples
    n_classes : int
        Number of distinct labels.
    """
    cdef np.ndarray[ndim=2, dtype=np.npy_intp, mode='c'] trans
    cdef np.npy_intp i

    trans = np.zeros((n_classes, n_classes), dtype=np.intp)

    for i in range(y.shape[0] - 1):
        trans[y[i], y[i + 1]] += 1
    return trans
