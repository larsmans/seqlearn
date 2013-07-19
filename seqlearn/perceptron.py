# Copyright 2013 Lars Buitinck

from __future__ import division

import numpy as np

from sklearn.preprocessing import LabelBinarizer, LabelEncoder

from .base import BaseSequenceClassifier
from ._decode import viterbi
from ._utils import atleast2d_or_csr, check_random_state, safe_sparse_dot


class StructuredPerceptron(BaseSequenceClassifier):
    """Structured perceptron for sequence classification.

    This implements the averaged structured perceptron algorithm of Collins,
    with the addition of a learning rate.

    References
    ----------
    M. Collins (2002). Discriminative training methods for hidden Markov
    models: Theory and experiments with perceptron algorithm. EMNLP.

    """
    def __init__(self, decode="viterbi", learning_rate=.1, max_iter=10,
                 random_state=None, verbose=0):
        self.decode = decode
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y, lengths):
        X = atleast2d_or_csr(X)
        y, Y_true = _one_hot(y)
        lengths = np.asarray(lengths)
        n_samples, n_features = X.shape
        n_classes = Y_true.shape[1]

        start = np.cumsum(lengths) - lengths
        end = start + lengths

        t_trans, t_init, t_final = _count_trans(y, start, end, n_classes)

        w = np.zeros((n_classes, n_features))
        b = np.zeros(n_classes)
        w_trans = np.zeros((n_classes, n_classes))
        w_init = np.zeros(n_classes)
        w_final = np.zeros(n_classes)

        w_avg = np.zeros(w.shape)
        b_avg = np.zeros(b.shape)
        w_trans_avg = np.zeros(w_trans.shape)
        w_init_avg = np.zeros(w_init.shape)
        w_final_avg = np.zeros(w_final.shape)

        lr = self.learning_rate

        sequence_ids = np.arange(lengths.shape[0])
        rng = check_random_state(self.random_state)

        decode = viterbi

        for it in xrange(self.max_iter):
            rng.shuffle(sequence_ids)

            sum_loss = 0

            for i in sequence_ids:
                Score = safe_sparse_dot(X[start[i]:end[i]], w.T) + b
                y_pred = decode(Score, w_trans, w_init, w_final)
                y_pred, Y_pred = _one_hot(y_pred)
                loss = (y_pred != y).sum()

                if loss:
                    sum_loss += loss

                    Y_t_i = Y_true[start[i]:end[i]]
                    Y_diff = Y_pred - Y_t_i

                    w -= lr * safe_sparse_dot(Y_diff.T, X)
                    b -= lr * Y_diff.sum(axis=0)

                    p_trans, p_init, p_final = (
                        _count_trans(y_pred, start, end, n_classes))
                    w_trans -= lr * (p_trans - t_trans)
                    w_init -= lr * (p_init - t_init)
                    w_final -= lr * (p_final - t_final)

                    w_avg += w
                    b_avg += b
                    w_trans_avg += w_trans
                    w_init_avg += w_init
                    w_final_avg += w_final

            if self.verbose:
                # XXX the loss reported is that for w, but the one for
                # w_avg is what matters for early stopping.
                print("Iteration %d, loss = %.3f" % (it + 1, loss / n_samples))

        self.coef_ = w_avg
        self.coef_ /= n_samples * it
        self.coef_init_ = w_init_avg
        self.coef_init_ /= n_samples * it
        self.coef_trans_ = w_trans_avg
        self.coef_trans_ /= n_samples * it
        self.coef_final_ = w_final_avg
        self.coef_final_ /= n_samples * it
        self.intercept_ = b_avg
        self.intercept_ /= n_samples * it

        return self


# TODO handle second-order transitions (trigrams)
def _count_trans(y, start, end, n_classes):
    """Count transitions in a target vector.

    Parameters
    ----------
    y : array, shape = n_samples
    n_classes : int
        Number of distinct labels.
    """
    #n_samples, = y.shape
    trans = np.zeros((n_classes, n_classes))
    inits = np.zeros(n_classes)
    final = np.zeros(n_classes)

    for i in xrange(len(start)):
        #start, end = offsets[i], offsets[i + 1]
        inits[y[start[i]]] += 1
        final[y[end[i] - 1]] += 1
        for j in xrange(start[i], end[i] - 1):
            trans[y[j], y[j + 1]] += 1

    return trans, inits, final


def _one_hot(y):
    y = LabelEncoder().fit_transform(y)
    Y = LabelBinarizer().fit_transform(y)
    if len(Y.shape) == 1:
        Y = np.atleast_2d(Y).T
        Y = np.hstack([1 - Y, Y])
    return y, Y
