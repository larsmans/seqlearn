import numpy as np
from numpy.testing import assert_array_equal
from scipy.sparse import csc_matrix

from seqlearn.crf import LinearChainCRF


def test_crf():
    X = np.array([[1, 0, 0],
                  [1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1],
                  [1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])

    X = csc_matrix(X)
    y = np.array(['0', '0', '1', '1', '0', '1', '1'])
    lengths = np.array([4, 3])

    for it in [1, 5, 10]:
        clf = LinearChainCRF(max_iter=it)
        clf.fit(X, y, lengths)

        y_pred = clf.predict(X, lengths)
        assert_array_equal(y, y_pred)
