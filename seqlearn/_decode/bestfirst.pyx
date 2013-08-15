# Copyright Lars Buitinck / University of Amsterdam 2013

cimport cython
cimport numpy as np
import numpy as np

np.import_array()

cdef np.float64_t NEGINF = -np.inf


@cython.boundscheck(False)
@cython.wraparound(False)
def bestfirst(np.ndarray[ndim=2, dtype=np.float64_t] score,
              trans_score,
              np.ndarray[ndim=2, dtype=np.float64_t] trans,
              np.ndarray[ndim=1, dtype=np.float64_t, mode="c"] init,
              np.ndarray[ndim=1, dtype=np.float64_t, mode="c"] final):
    """First-order heuristic best-first decoder.

    See viterbi for the arguments. score may be overwritten.
    trans_score is not supported yet.
    """

    cdef:
        np.ndarray[ndim=1, dtype=np.npy_intp, mode="c"] path
        np.float64_t candidate, maxval
        np.npy_intp i, j, maxind, n_samples, n_states

    if trans_score is not None:
        raise NotImplementedError("No transition scores for bestfirst yet.")

    n_samples, n_states = score.shape[0], score.shape[1]

    path = np.empty(n_samples, dtype=np.intp)
    score[0] += init
    score[n_samples - 1] += final
    path[0] = np.argmax(score[0])

    for i in range(1, n_samples):
        maxind = 0
        maxval = NEGINF
        for j in range(n_states):
            candidate = trans[path[i - 1], j] + score[i, j]
            if candidate > maxval:
                maxind = j
                maxval = candidate

        path[i] = maxind

    return path
