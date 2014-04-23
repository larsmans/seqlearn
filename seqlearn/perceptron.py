# Copyright 2013 Lars Buitinck
# encoding: utf-8

from __future__ import division, print_function, absolute_import

import sys

import numpy as np
from scipy.sparse import csc_matrix
from sklearn.externals import six

from .base import BaseSequenceClassifier
from ._utils import (atleast2d_or_csr, check_random_state, count_trans,
                     make_trans_matrix, safe_add, safe_sparse_dot)

class StructuredPerceptron(BaseSequenceClassifier):
    """Structured perceptron for sequence classification.

    This implements the averaged structured perceptron algorithm of Collins
    and Daumé, with the addition of an adaptive learning rate.

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

    trans_features : boolean, optional
        Whether to attach features to transitions between labels as well as
        individual labels. This requires more time, more memory and more
        samples to train properly.

    verbose : integer, optional
        Verbosity level. Defaults to zero (quiet mode).

    References
    ----------
    M. Collins (2002). Discriminative training methods for hidden Markov
    models: Theory and experiments with perceptron algorithms. EMNLP.

    Hal Daumé III (2006). Practical Structured Learning Techniques for
    Natural Language Processing. Ph.D. thesis, U. Southern California.

    """
    def __init__(self, decode="viterbi", lr_exponent=.1, max_iter=10,
                 random_state=None, trans_features=False, verbose=0):
        self.decode = decode
        self.lr_exponent = lr_exponent
        self.max_iter = max_iter
        self.random_state = random_state
        self.trans_features = trans_features
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
        n_classes = len(classes)

        if self.trans_features:
            trans_true = make_trans_matrix(y, n_classes)
        class_range = np.arange(n_classes)
        Y_true = y.reshape(-1, 1) == class_range

        lengths = np.asarray(lengths)
        n_samples, n_features = X.shape

        end = np.cumsum(lengths)
        start = end - lengths

        w = np.zeros((n_classes, n_features), order='F')
        b_trans = np.zeros((n_classes, n_classes))
        b_init = np.zeros(n_classes)
        b_final = np.zeros(n_classes)

        w_avg = np.zeros_like(w)
        b_trans_avg = np.zeros_like(b_trans)
        b_init_avg = np.zeros_like(b_init)
        b_final_avg = np.zeros_like(b_final)

        if self.trans_features:
            w_trans = np.zeros((n_classes, n_classes, n_features), order='F')
            w_t_flat = w_trans.reshape(-1, n_features)
            w_trans_avg = np.zeros_like(w_trans)
            w_t_avg_flat = w_trans_avg.reshape(-1, n_features)

        sequence_ids = np.arange(lengths.shape[0])
        rng = check_random_state(self.random_state)

        avg_count = 1.
        lr_exponent = self.lr_exponent

        for it in six.moves.xrange(1, self.max_iter + 1):
            lr = 1. / (it ** lr_exponent)

            if self.verbose:
                print("Iteration {0:2d}".format(it), end="... ")
                sys.stdout.flush()

            rng.shuffle(sequence_ids)

            sum_loss = 0

            for i in sequence_ids:
                X_i = X[start[i]:end[i]]
                score = safe_sparse_dot(X_i, w.T)
                if self.trans_features:
                    trans_score = safe_sparse_dot(X_i, w_t_flat.T)
                    trans_score = trans_score.reshape(-1, n_classes, n_classes)
                else:
                    trans_score = None
                y_pred = decode(score, trans_score, b_trans, b_init, b_final)
                y_t_i = y[start[i]:end[i]]
                loss = (y_pred != y_t_i).sum()

                if loss:
                    sum_loss += loss

                    Y_t_i = Y_true[start[i]:end[i]]
                    Y_pred = y_pred.reshape(-1, 1) == class_range
                    Y_pred = Y_pred.astype(np.float64)

                    Y_diff = csc_matrix(Y_pred - Y_t_i)
                    Y_diff *= -lr
                    w_update = safe_sparse_dot(Y_diff.T, X_i)

                    if self.trans_features:
                        trans_pred = make_trans_matrix(y_pred, n_classes)

                        trans_diff = trans_true[start[i]:end[i]].toarray()
                        safe_add(trans_diff, -trans_pred)
                        trans_diff = csc_matrix(trans_diff)
                        trans_diff *= -lr

                        w_trans_update = safe_sparse_dot(trans_diff.T, X_i)
                        safe_add(w_t_flat, w_trans_update)

                        w_trans_update *= avg_count
                        safe_add(w_t_avg_flat, w_trans_update)

                    t_trans = count_trans(y_t_i, n_classes)
                    p_trans = count_trans(y_pred, n_classes)
                    b_trans_update = lr * (p_trans - t_trans)
                    b_init_update = lr * (Y_pred[0] - Y_true[start[i]])
                    b_final_update = lr * (Y_pred[-1] - Y_true[end[i] - 1])

                    safe_add(w, w_update)
                    b_trans -= b_trans_update
                    b_init -= b_init_update
                    b_final -= b_final_update

                    w_update *= avg_count
                    b_trans_update *= avg_count
                    b_init_update *= avg_count
                    b_final_update *= avg_count

                    safe_add(w_avg, w_update)
                    b_trans_avg -= b_trans_update
                    b_init_avg -= b_init_update
                    b_final_avg -= b_final_update

            if self.verbose:
                # XXX the loss reported is that for w, but the one for
                # w_avg is what matters for early stopping.
                print("loss = {0:.4f}".format(sum_loss / n_samples))

            avg_count += 1.

        w -= w_avg / avg_count
        if self.trans_features:
            w_trans -= w_trans_avg / avg_count
        b_init -= b_init_avg / avg_count
        b_trans -= b_trans_avg / avg_count
        b_final -= b_final_avg / avg_count

        self.coef_ = w
        if self.trans_features:
            self.coef_trans_ = w_trans
        self.intercept_init_ = b_init
        self.intercept_trans_ = b_trans
        self.intercept_final_ = b_final

        self.classes_ = classes

        return self
