# Copyright Lars Buitinck 2013.

"""Decoding (inference) algorithms."""

import numpy as np


def bestfirst(Phi, trans, init, final):
    """First-order heuristic best-first decoder."""

    n_samples, _ = Phi.shape

    path = np.empty(n_samples, dtype=np.int32)
    path[0] = np.argmax(init + Phi[0])

    for i in xrange(1, n_samples - 1):
        path[i] = np.argmax(trans[path[i - 1], :] + Phi[i])

    path[-1] = np.argmax(trans[path[-2], :] + Phi[-1] + final)

    return path


def viterbi(Phi, trans, init, final):
    """First-order Viterbi algorithm.

    Parameters
    ----------
    Phi : array, shape = (n_samples, n_states)
        Scores per sample/class combination.
        In a linear model, this is X * w.T.
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

    n_samples, n_states = Phi.shape
    n_states = trans.shape[0]

    dp = np.empty((n_samples, n_states))
    dp[0, :] = init + Phi[0, :]

    backp = np.empty((n_samples, n_states), dtype=np.int32)
    _idx = np.mgrid[0:n_states]

    for i in xrange(1, n_samples):
        candidates = trans.T + Phi[i, :].reshape((n_states, 1)) + dp[i - 1, :]
        backp_i = np.argmax(candidates, axis=1)
        backp[i, :] = backp_i
        dp[i, :] = candidates[_idx, backp_i]

    dp[-1, :] += final

    # Path backtracking
    path = np.empty(n_samples, dtype=np.int32)
    path[-1] = dp[-1, :].argmax()

    for i in xrange(n_samples - 2, -1, -1):
        path[i] = backp[i + 1, path[i + 1]]

    return path


DECODERS = {"bestfirst": bestfirst,
            "viterbi": viterbi}
