from numpy.testing import assert_array_equal

import numpy as np

from seqlearn.viterbi import viterbi

def test_viterbi():
    # Example taken from Wikipedia. Samples can be "normal", "cold" or "dizzy"
    # (represented as one-hot feature vectors). States are "Healthy" and
    # "Fever". ['normal', 'cold', 'dizzy'] has most optimal state sequence
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
    Phi = np.dot(X, w.T)

    assert_array_equal(viterbi(Phi, trans, start, final), [0, 0, 1])
