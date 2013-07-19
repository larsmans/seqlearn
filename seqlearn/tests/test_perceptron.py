from numpy.testing import assert_array_equal

from seqlearn.perceptron import StructuredPerceptron


def test_perceptron():
    X = [[0, 1, 0],
         [0, 1, 0],
         [1, 0, 0],
         [0, 1, 0],
         [0, 0, 1],
         [0, 0, 1],
         [0, 1, 0],
         [1, 0, 0],
         [1, 0, 0]]

    y = [0, 0, 0, 0, 1, 1, 0, 2, 2]

    clf = StructuredPerceptron(verbose=True).fit(X, y, [len(y)])
    assert_array_equal(y, clf.predict(X))
