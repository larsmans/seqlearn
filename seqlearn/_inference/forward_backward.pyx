# Author: Chyi-Kwei Yau

"""Forward-Backward algorithm for CRF training & posterior calculation"""

cimport cython
cimport numpy as np
import numpy as np

from .._utils import logsumexp


np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
def _forward(np.ndarray[ndim=2, dtype=np.float64_t] score,
            np.ndarray[ndim=3, dtype=np.float64_t] trans_score,
            np.ndarray[ndim=2, dtype=np.float64_t] b_trans,
            np.ndarray[ndim=1, dtype=np.float64_t] init,
            np.ndarray[ndim=1, dtype=np.float64_t] final):
    """
    Forward Algorithm

    Parameters
    ----------
    score : array, shape = (n_samples, n_states)
        Scores per sample/class combination; in a linear model, X * w.T.
        May be overwritten.
    trans_score : array, shape = (n_samples, n_states, n_states), optional
        Scores per sample/transition combination.
    b_trans : array, shape = (n_states, n_states)
        Transition weights.
    init : array, shape = (n_states,)
    final : array, shape = (n_states,)
        Initial and final state weights.

    Return
    ------
    forward : array, shape = (n_samples, n_states)

    References
    ----------
    L. R. Rabiner (1989). A tutorial on hidden Markov models and selected
    applications in speech recognition. Proc. IEEE 77(2):257-286.

    """

    cdef np.ndarray[ndim=2, dtype=np.float64_t] forward
    cdef np.npy_intp i, j, k, m, n_samples, n_states, last_index

    n_samples, n_states = score.shape[0], score.shape[1]
    last_index = n_samples - 1
    forward = np.empty((n_samples, n_states), dtype=np.float64)

    # initialize
    for j in range(n_states):
        forward[0, j] = init[j] + score[0, j]

    for i in range(1, n_samples):
        for k in range(n_states):
            temp_array = forward[i-1, :] + b_trans[:, k] + score[i, k]
            #if trans_score is not None:
            #    temp_array += trans_score[i-1, k, :]
            if i == last_index:
                temp_array += final[k]
            forward[i, k] = logsumexp(temp_array)

    return forward


@cython.boundscheck(False)
@cython.wraparound(False)
def _backward(np.ndarray[ndim=2, dtype=np.float64_t] score,
            np.ndarray[ndim=3, dtype=np.float64_t] trans_score,
            np.ndarray[ndim=2, dtype=np.float64_t] b_trans,
            np.ndarray[ndim=1, dtype=np.float64_t] init,
            np.ndarray[ndim=1, dtype=np.float64_t] final):
    
    """
    Backward Algorithm (similar to forward Algorithm)

    Parameters
    ----------
    Same as Forward function


    Returns
    -------
    backward : array, shape = (n_samples, n_states)

    """

    cdef np.ndarray[ndim=2, dtype=np.float64_t] backward
    cdef np.npy_intp i, j, k, m, n_samples, n_states, last_index

    n_samples, n_states = score.shape[0], score.shape[1]
    last_index = n_samples - 1

    backward = np.empty((n_samples, n_states), dtype=np.float64)
    
    # initialize
    for j in range(n_states):
        backward[last_index, j] = final[j]

    for i in range(last_index-1, -1, -1):
        for k in range(n_states):
            temp_array = backward[i+1, :] + b_trans[k, :] + score[i+1, :]
            #if trans_score is not None:
            #    temp_array += trans_score[i, :, k]
            print temp_array
            backward[i, k] = logsumexp(temp_array)

    return backward


@cython.boundscheck(False)
@cython.wraparound(False)
def _posterior(np.ndarray[ndim=2, dtype=np.float64_t] score,
            np.ndarray[ndim=3, dtype=np.float64_t] trans_score,
            np.ndarray[ndim=2, dtype=np.float64_t] b_trans,
            np.ndarray[ndim=1, dtype=np.float64_t] init,
            np.ndarray[ndim=1, dtype=np.float64_t] final):
    
    """
    Calculate posterior distrubtion based on Forward-Backward algorithm

    Parameters
    ----------
    Same as Forward function
    """

    cdef np.ndarray[ndim=2, dtype=np.float64_t] forward, backward, state_posterior, trans_posterior
    cdef np.npy_intp i, j, k, n_samples, n_states
    # log likelihood value
    cdef np.float64_t ll, temp_trans_val

    n_samples, n_states = score.shape[0], score.shape[1]

    # initialize
    state_posterior = np.empty((n_samples, n_states), dtype=np.float64)
    trans_posterior = np.zeros((n_states, n_states), dtype=np.float64)

    # get forward-backward values
    forward = _forward(score, trans_score, b_trans, init, final)
    backward = _backward(score, trans_score, b_trans, init, final)

    # get log likelihood
    ll = logsumexp(forward[n_samples-1, :] + final)

    # states poterior
    for i in range(n_samples):
        for j in range(n_states):
            state_posterior[i, j] = forward[i, j] + backward[i, j] - ll
    state_posterior = np.exp(state_posterior)

    # transition posterior
    for i in range(n_samples-1):
        for j in range(n_states):
            for k in range(n_states):
                temp_trans_val = forward[i, j] + b_trans[j, k] + score[i+1, k] + backward[i+1, k] - ll
                #if trans_score is not None:
                #    temp_trans_val += trans_score[i, j, k]
                trans_posterior[j, k] += np.exp(temp_trans_val)

    return state_posterior, trans_posterior

