# Copyright Lars Buitinck 2013.

import numpy as np


def bestfirst(Score, trans, init, final):
    """First-order heuristic best-first decoder."""

    n_samples, _ = Score.shape

    path = np.empty(n_samples, dtype=np.intp)
    path[0] = np.argmax(init + Score[0])

    for i in xrange(1, n_samples - 1):
        path[i] = np.argmax(trans[path[i - 1], :] + Score[i])

    path[-1] = np.argmax(trans[path[-2], :] + Score[-1] + final)

    return path
