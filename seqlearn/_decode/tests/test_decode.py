from nose.tools import assert_greater
from numpy.testing import assert_array_equal

import numpy as np
from sklearn.metrics import accuracy_score

from seqlearn._decode import bestfirst, viterbi


def test_wikipedia_example():
    # HMM example taken from Wikipedia. Samples can be "normal", "cold" or
    # "dizzy" (represented as one-hot feature vectors). States are "Healthy"
    # and "Fever". ['normal', 'cold', 'dizzy'] has optimal state sequence
    # ['Healthy', 'Healthy', 'Fever'].

    start = np.log([.6, .4])
    final = np.log([.5, .5])    # not given, so assume uniform probabilities

    trans = np.log([[.7, .3],
                    [.4, .6]])

    w = np.log([[.5, .4, .1],
                [.1, .3, .6]])
    X = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])
    score = np.dot(X, w.T)

    assert_array_equal(bestfirst(score, None, trans, start, final), [0, 0, 1])
    assert_array_equal(viterbi(score, None, trans, start, final), [0, 0, 1])


def test_dna():
    # HMM example taken from Borodovsky and Ekisheva (2006), Problems and
    # Solutions in Biological Sequence Analysis, p. 80.
    # Four (one-hot) features T, C, A and G, two states H and L
    # (high and low C+G content).

    start = np.log([.5, .5])
    final = start

    trans = np.log([[.5, .5],
                    [.4, .6]])

    # XXX in a binary problem, w of shape (n_features,) should be enough
    w = np.log([[.2, .3, .2, .3],
                [.3, .2, .3, .2]])
    X = np.array([[0, 0, 0, 1],     # G
                  [0, 0, 0, 1],     # G
                  [0, 1, 0, 0],     # C
                  [0, 0, 1, 0],     # A
                  [0, 1, 0, 0],     # C
                  [1, 0, 0, 0],     # T
                  [0, 0, 0, 1],     # G
                  [0, 0, 1, 0],     # A
                  [0, 0, 1, 0]])    # A
    score = np.dot(X, w.T)

    # HHHLLLLLL
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1])

    assert_array_equal(viterbi(score, None, trans, start, final), y_true)

    # For this problem, Viterbi actually is better than best-first.
    bf = bestfirst(score, None, trans, start, final)
    assert_greater(accuracy_score(y_true, bf), .75)
