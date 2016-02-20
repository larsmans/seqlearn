from numpy.testing import assert_array_equal
from numpy.testing import assert_raises

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

def test_perceptron_mask():
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
    
    trans_constraints = [('spam','eggs'), ('spam', 'ham')]

    clf = StructuredPerceptron(verbose=True, random_state=42, max_iter=15,
                               trans_constraints=trans_constraints)

    # Try again with string labels and sparse input.
    y_str = np.array(["eggs", "ham", "spam"])[y]

    
    clf.fit(csc_matrix(X), y_str, [len(y_str)])
    
    # Still fits
    assert_array_equal(y_str, clf.predict(coo_matrix(X)))
    # Weights are overridden properly
    assert_array_equal([clf.intercept_trans_[2,0], clf.intercept_trans_[2,1]], 
                       [clf.CONSTRAINT_VALUE]*2)
                       
    # Add impossible constriants and model should fail to converge
    impossible_constraints = [('spam','eggs'), ('eggs', 'ham')]
    clf2 = StructuredPerceptron(verbose=True, random_state=12, max_iter=15,
                               trans_constraints=impossible_constraints)
    
    clf2.fit(csc_matrix(X), y_str, [len(y_str)])
    
    # Should raise error saying that prediction is incorrect
    assert_raises(AssertionError, assert_array_equal, y_str, clf2.predict(coo_matrix(X)))