from numpy.testing import assert_array_equal

import numpy as np
from sklearn.base import clone

from seqlearn.perceptron import StructuredPerceptron


def test_perceptron():
    X = [[0, 1, 0],
         [0, 1, 0],
         [1, 0, 0],
         [0, 1, 0],
         [1, 0, 0],
         [0, 0, 1],
         [0, 0, 1],
         [0, 1, 0],
         [1, 0, 0],
         [1, 0, 0]]

    y = [0, 0, 0, 0, 0, 1, 1, 0, 2, 2]

    clf = StructuredPerceptron(verbose=False, random_state=37, max_iter=15)
    clf.fit(X, y, [len(y)])
    assert_array_equal(y, clf.predict(X))

    # try again, with string labels
    y = np.array(["eggs", "ham", "spam"])[y]

    clf = clone(clf)
    clf.fit(X, y, [len(y)])
    assert_array_equal(y, clf.predict(X))

    X2 = np.vstack([X, X])
    y2 = np.hstack([y, y])
    assert_array_equal(y2, clf.predict(X2, lengths=[len(y), len(y)]))


def test_perceptron_single_iter():
    """Assert that averaging works after a single iteration."""
    clf = StructuredPerceptron(max_iter=1)
    clf.fit([[1, 2, 3]], [1], [1])  # no exception
