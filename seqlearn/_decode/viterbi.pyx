# Copyright Lars Buitinck 2013.

"""Decoding (inference) algorithms."""

cimport cython
cimport numpy as np
import numpy as np

np.import_array()

cdef np.float64_t NEGINF = -np.inf


@cython.boundscheck(False)
@cython.wraparound(False)
def viterbi(np.ndarray[ndim=2, dtype=np.float64_t] score,
            np.ndarray[ndim=3, dtype=np.float64_t] trans_score,
            np.ndarray[ndim=2, dtype=np.float64_t] b_trans,
            np.ndarray[ndim=1, dtype=np.float64_t] init,
            np.ndarray[ndim=1, dtype=np.float64_t] final):
    """First-order Viterbi algorithm.

    Parameters
    ----------
    score : array, shape = (n_samples, n_states)
        Scores per sample/class combination; in a linear model, X * w.T.
        May be overwritten.
    trans_score : array, shape = (n_samples, n_states, n_states), optional
        Scores per sample/transition combination.
    trans : array, shape = (n_states, n_states)
        Transition weights.
    init : array, shape = (n_states,)
    final : array, shape = (n_states,)
        Initial and final state weights.

    References
    ----------
    L. R. Rabiner (1989). A tutorial on hidden Markov models and selected
    applications in speech recognition. Proc. IEEE 77(2):257-286.
    """

    cdef np.ndarray[ndim=2, dtype=np.npy_intp, mode='c'] backp
    cdef np.ndarray[ndim=1, dtype=np.npy_intp, mode='c'] path
    cdef np.float64_t candidate, maxval
    cdef np.npy_intp i, j, k, n_samples, n_states

    n_samples, n_states = score.shape[0], score.shape[1]

    backp = np.empty((n_samples, n_states), dtype=np.intp)

    for j in range(n_states):
        score[0, j] += init[j]

    # Forward recursion. score is reused as the DP table.
    for i in range(1, n_samples):
        for k in range(n_states):
            maxind = 0
            maxval = NEGINF
            for j in range(n_states):
                candidate = score[i - 1, j] + b_trans[j, k] + score[i, k]
                if trans_score is not None:
                    candidate += trans_score[i, j, k]
                if candidate > maxval:
                    maxind = j
                    maxval = candidate

            score[i, k] = maxval
            backp[i, k] = maxind

    for j in range(n_states):
        score[n_samples - 1, j] += final[j]

    # Path backtracking
    path = np.empty(n_samples, dtype=np.intp)
    path[n_samples - 1] = score[n_samples - 1, :].argmax()

    for i in range(n_samples - 2, -1, -1):
        path[i] = backp[i + 1, path[i + 1]]

    return path
