from numpy.testing import assert_array_equal

import numpy as np
from scipy.sparse import csr_matrix

from seqlearn._utils import safe_add


def test_safe_add():
    X1 = np.zeros((4, 13), dtype=np.float64)
    X2 = X1.copy()
    Y = csr_matrix(np.arange(4 * 13, dtype=np.float64).reshape(4, 13) % 4)

    X1 += Y
    safe_add(X2, Y)
    assert_array_equal(X1, X2)

    X = np.zeros((13, 4), dtype=np.float64)
    YT = Y.T
    safe_add(X, YT)
    assert_array_equal(X1, X.T)
