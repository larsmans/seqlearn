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
