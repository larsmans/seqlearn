from numpy.testing import assert_array_equal

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
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

    # Try again with string labels and sparse input.
    y_str = np.array(["eggs", "ham", "spam"])[y]

    clf = clone(clf)
    clf.fit(csc_matrix(X), y_str, [len(y_str)])
    assert_array_equal(y_str, clf.predict(coo_matrix(X)))

    X2 = np.vstack([X, X])
    y2 = np.hstack([y_str, y_str])
    assert_array_equal(y2, clf.predict(X2, lengths=[len(y), len(y)]))

    # Train with Viterbi, test with best-first to make StructuredPerceptron
    # behave a bit more like a linear model.
    # DISABLED: this test is unstable.
    #clf.fit(X, y, [len(y)])
    #clf.set_params(decode="bestfirst")
    #y_linearmodel = np.dot(X, clf.coef_.T).argmax(axis=1)
    #assert_array_equal(clf.predict(X), y_linearmodel)


def test_perceptron_single_iter():
    """Assert that averaging works after a single iteration."""
    clf = StructuredPerceptron(max_iter=1)
    clf.fit([[1, 2, 3]], [1], [1])  # no exception
