# Linear Chain CRF with Stochastic Gradient Descent.
# author: Chyi-Kwei Yau, 2014

from __future__ import division, print_function

import numpy as np

from .base import BaseSequenceClassifier
from ._utils import (atleast2d_or_csr, check_random_state, count_trans,
                     safe_add, safe_sparse_dot)

from ._inference.forward_backward import _posterior


class LinearChainCRF(BaseSequenceClassifier):
    """Linear Chian Conditional Random Field for sequence classification.

    This implements a linear chain CRF with Stochastic Gradient Descent.

    Parameters
    ----------
    decode : string, optional
        Decoding algorithm, either "bestfirst" or "viterbi" (default).

    lr : float, optional
        Initial learning rate

    lr_exponent : float, optional
        Exponent for inverse scaling learning rate. The effective learning
        rate is lr / (t ** lr_exponent), where t is the iteration number.

    max_iter : integer, optional
        Number of iterations (aka. epochs). Each sequence is visited once in
        each iteration.

    random_state : {integer, np.random.RandomState}, optional
        Random state or seed used for shuffling sequences within each
        iteration.

    reg: L2 regularization value

    compute_obj_val: compute objective value. Set this to True to check whether the objective
        value converges.

    verbose : integer, optional
        Verbosity level. Defaults to zero (quiet mode).

    References
    ----------
    J. Lafferty (2001). Conditional random fields: Probabilistic models 
    for segmenting and labeling sequence data

    C. Sutton (2006) An Introduction to Conditional Random Fields for
    Relational Learning

    N. Schraudolph (2006). Accelerated Training of Conditional Random
    Fields with Stochastic Gradient Methods

    """

    def __init__(self, decode="viterbi", lr=1.0, lr_exponent=.1, max_iter=10,
                 random_state=None, reg=.01, compute_obj_val=False, verbose=0):
        self.decode = decode
        self.lr = lr
        self.lr_exponent = lr_exponent
        self.max_iter = max_iter
        self.random_state = random_state
        self.reg = reg
        self.compute_obj_val = compute_obj_val
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
        self : LinearChainCRF
        """

        X = atleast2d_or_csr(X)

        classes, y = np.unique(y, return_inverse=True)
        n_classes = len(classes)

        class_range = np.arange(n_classes)
        Y_true = y.reshape(-1, 1) == class_range

        lengths = np.asarray(lengths)
        n_samples, n_features = X.shape
        n_sequence = lengths.shape[0]

        end = np.cumsum(lengths)
        start = end - lengths

        # initialize parameters
        w = np.zeros((n_classes, n_features), order='F')
        b_trans = np.zeros((n_classes, n_classes))
        b_init = np.zeros(n_classes)
        b_final = np.zeros(n_classes)

        w_avg = np.zeros_like(w)
        b_trans_avg = np.zeros_like(b_trans)
        b_init_avg = np.zeros_like(b_init)
        b_final_avg = np.zeros_like(b_final)

        sequence_ids = np.arange(n_sequence)
        rng = check_random_state(self.random_state)

        avg_count = 1.
        for it in xrange(1, self.max_iter + 1):
            if self.verbose:
                print("Iteration {0:2d}...".format(it))

            if self.compute_obj_val:
                sample_count = 0
                sum_obj_val = 0.0

            rng.shuffle(sequence_ids)

            lr = self.lr / (it ** self.lr_exponent)

            reg = self.reg / n_sequence

            for i in sequence_ids:
                X_i = X[start[i]:end[i]]
                y_t_i = Y_true[start[i]:end[i]]
                t_trans = count_trans(y[start[i]:end[i]], n_classes)

                score = safe_sparse_dot(X_i, w.T)

                # posterior distribution for states & transtion
                post_state, post_trans, ll = _posterior(score, None, b_trans, b_init, b_final)

                if self.compute_obj_val:
                    w_true = safe_sparse_dot(y_t_i.T, X_i)
                    feature_val = np.sum(w_true * w)
                    trans_val = np.sum(t_trans * b_trans)
                    init_val = np.sum(y_t_i[0] * b_init)
                    final_val = np.sum(y_t_i[-1] * b_final)
                    sum_obj_val += feature_val + trans_val + init_val + final_val - ll - (0.5 * reg * np.sum(w * w))

                    sample_count += 1
                    if sample_count % 1000 == 0:
                        avg_obj_val = sum_obj_val / sample_count
                        print("iter: {0:d}, sample: {1:d}, avg. objective value {2:.4f}".format(
                            it, sample_count, avg_obj_val))

                # update feature w
                w_update = safe_sparse_dot(lr * (y_t_i - post_state).T, X_i) - ((lr * reg) * w)

                # update init & final matrix
                b_init_update = lr * (post_state[0, :] - y_t_i[0] + reg * b_init)
                b_final_update = lr * (post_state[-1, :] - y_t_i[-1] + reg * b_final)
                
                # update transition matrix
                b_trans_update = lr * (post_trans - t_trans + reg * b_trans)

                safe_add(w, w_update)
                b_init -= b_init_update
                b_final -= b_final_update
                b_trans -= b_trans_update

                w_update *= avg_count
                b_trans_update *= avg_count
                b_init_update *= avg_count
                b_final_update *= avg_count

                safe_add(w_avg, w_update)
                b_trans_avg -= b_trans_update
                b_init_avg -= b_init_update
                b_final_avg -= b_final_update

            avg_count += 1.

        w -= w_avg / avg_count
        b_init -= b_init_avg / avg_count
        b_trans -= b_trans_avg / avg_count
        b_final -= b_final_avg / avg_count

        self.coef_ = w
        self.intercept_init_ = b_init
        self.intercept_trans_ = b_trans
        self.intercept_final_ = b_final
        self.classes_ = classes

        return self
