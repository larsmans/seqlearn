# Copyright 2013 Lars Buitinck

from __future__ import division, print_function

import sys

import numpy as np
from scipy.sparse import csr_matrix

from .base import BaseSequenceClassifier
from ._utils import (atleast2d_or_csr, check_random_state, count_trans,
                     safe_add, safe_sparse_dot)


class StructuredPerceptron(BaseSequenceClassifier):
    """Structured perceptron for sequence classification.

    This implements the averaged structured perceptron algorithm of Collins,
    with the addition of an adaptive learning rate.

    Parameters
    ----------
    decode : string, optional
        Decoding algorithm, either "bestfirst" or "viterbi" (default).

    lr_exponent : float, optional
        Exponent for inverse scaling learning rate. The effective learning
        rate is 1. / (t ** lr_exponent), where t is the iteration number.

    max_iter : integer, optional
        Number of iterations (aka. epochs). Each sequence is visited once in
        each iteration.

    random_state : {integer, np.random.RandomState}, optional
        Random state or seed used for shuffling sequences within each
        iteration.

    verbose : integer, optional
        Verbosity level. Defaults to zero (quiet mode).

    References
    ----------
    M. Collins (2002). Discriminative training methods for hidden Markov
    models: Theory and experiments with perceptron algorithm. EMNLP.

    """
    def __init__(self, decode="viterbi", lr_exponent=.1, max_iter=10,
                 random_state=None, verbose=0):
        self.decode = decode
        self.lr_exponent = lr_exponent
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y, lengths):
        """Fit to a set of sequences.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Feature matrix of individual samples.

        y : array-like, shape (n_samples,)
            Target labels.

        lengths : array-like of integers, shape (n_sequences,)
            Lengths of the individual sequences in X, y. The sum of these
            should be n_samples.

        Returns
        -------
        self : StructuredPerceptron
        """

        decode = self._get_decoder()

        X = atleast2d_or_csr(X)

        classes, y = np.unique(y, return_inverse=True)
        class_range = np.arange(len(classes))
        Y_true = y.reshape(-1, 1) == class_range

        lengths = np.asarray(lengths)
        n_samples, n_features = X.shape
        n_classes = Y_true.shape[1]

        end = np.cumsum(lengths)
        start = end - lengths

        w = np.zeros((n_classes, n_features), order='F')
        w_trans = np.zeros((n_classes, n_classes))
        w_init = np.zeros(n_classes)
        w_final = np.zeros(n_classes)

        w_avg = np.zeros_like(w)
        w_trans_avg = np.zeros_like(w_trans)
        w_init_avg = np.zeros_like(w_init)
        w_final_avg = np.zeros_like(w_final)

        sequence_ids = np.arange(lengths.shape[0])
        rng = check_random_state(self.random_state)

        lr_exponent = self.lr_exponent

        for it in xrange(1, self.max_iter + 1):
            lr = 1. / (it ** lr_exponent)

            if self.verbose:
                print("Iteration {0:2d}".format(it), end="... ")
                sys.stdout.flush()

            rng.shuffle(sequence_ids)

            sum_loss = 0

            for i in sequence_ids:
                X_i = X[start[i]:end[i]]
                Score = safe_sparse_dot(X_i, w.T)
                y_pred = decode(Score, w_trans, w_init, w_final)
                y_t_i = y[start[i]:end[i]]
                loss = (y_pred != y_t_i).sum()

                if loss:
                    sum_loss += loss

                    Y_t_i = Y_true[start[i]:end[i]]
                    Y_pred = y_pred.reshape(-1, 1) == class_range
                    Y_pred = Y_pred.astype(np.float64)

                    Y_diff = csr_matrix(Y_pred - Y_t_i)
                    Y_diff *= -lr
                    D = safe_sparse_dot(Y_diff.T, X_i)
                    safe_add(w, D)

                    t_trans = count_trans(y_t_i, n_classes)
                    p_trans = count_trans(y_pred, n_classes)
                    w_trans -= lr * (p_trans - t_trans)
                    w_init -= lr * (Y_pred[0] - Y_true[start[i]])
                    w_final -= lr * (Y_pred[-1] - Y_true[end[i] - 1])

                w_avg += w
                w_trans_avg += w_trans
                w_init_avg += w_init
                w_final_avg += w_final

            if self.verbose:
                # XXX the loss reported is that for w, but the one for
                # w_avg is what matters for early stopping.
                print("loss = {0:.4f}".format(sum_loss / n_samples))

        self.coef_ = w_avg
        self.coef_ /= it * len(lengths)
        self.coef_init_ = w_init_avg
        self.coef_init_ /= it * len(lengths)
        self.coef_trans_ = w_trans_avg
        self.coef_trans_ /= it * len(lengths)
        self.coef_final_ = w_final_avg
        self.coef_final_ /= it * len(lengths)

        self.classes_ = classes

        return self
