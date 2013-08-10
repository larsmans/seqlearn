from nose.tools import assert_equal
from numpy.testing import assert_array_equal

import numpy as np
from scipy.sparse import csr_matrix

from seqlearn._utils import make_trans_matrix


def test_trans_matrix():
    n_samples = 11
    n_classes = 5
    n_features = 14
    y = np.arange(n_samples) % n_classes
    Y = make_trans_matrix(y, n_classes, dtype=int)

    # XXX this test is far too simple
    assert_equal(Y.shape, (n_samples, n_classes ** 2))
