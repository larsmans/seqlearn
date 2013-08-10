# Copyright Lars Buitinck 2013.

import numpy as np


def bestfirst(score, trans_score, trans, init, final):
    """First-order heuristic best-first decoder."""

    if trans_score is not None:
        raise NotImplementedError("No transition scores for bestfirst yet.")

    n_samples, _ = score.shape

    path = np.empty(n_samples, dtype=np.intp)
    path[0] = np.argmax(init + score[0])

    for i in xrange(1, n_samples - 1):
        path[i] = np.argmax(trans[path[i - 1], :] + score[i])

    path[-1] = np.argmax(trans[path[-2], :] + score[-1] + final)

    return path
