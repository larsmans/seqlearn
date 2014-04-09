import numpy as np

from numpy.testing import assert_almost_equal
from seqlearn._inference.forward_backward import _forward, _backward, _posterior


init = np.array([.2, .1])
final = np.array([.1, .2])

trans = np.array([[.1, .2],
                  [.4, .3]])

score = np.array([[.3, .2],
                  [.4, .1],
                  [.5, .4]])


def test_forward():
    forward = _forward(score, None, trans, init, final)

    true_forward = np.array([[0.5, 0.3],
                             [1.7443, 1.4444],
                             [3.1375, 3.1425]])

    # assert equal
    assert_almost_equal(true_forward, forward, decimal=3)


def test_backward():
    backward = _backward(score, None, trans, init, final)
    
    #true value
    true_backward =  np.array([[2.6375, 2.8425],
                               [1.4444, 1.6444],
                               [0.0, 0.0]])
    # assert equal
    assert_almost_equal(true_backward, backward, decimal=3)


def test_forward_backward():

    forward = _forward(score, None, trans, init, final)
    backward = _backward(score, None, trans, init, final)

    assert_almost_equal(forward[-1, :], backward[0, :] + score[0, :] + init)

def test_posterior():
    pass
    #assert_almost_equal