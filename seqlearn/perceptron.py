# Copyright 2013 Lars Buitinck

from __future__ import division, print_function, absolute_import

import sys
from collections import namedtuple
import numpy as np

from .base import BaseSequenceClassifier
from ._utils import (atleast2d_or_csr, check_random_state, count_trans,
                     safe_sparse_dot)
from ._utils.shards import SequenceShards

_EpochState = namedtuple('EpochState', [
                'w', 'w_trans', 'w_init', 'w_final',
                'w_avg', 'w_trans_avg', 'w_init_avg', 'w_final_avg'])

class EpochState(_EpochState):
    def copy(self):
        return self.__class__(*[np.copy(w) for w in self])

    @classmethod
    def initial(cls, n_classes, n_features):
        w = np.zeros((n_classes, n_features), order='F')
        w_trans = np.zeros((n_classes, n_classes))
        w_init = np.zeros(n_classes)
        w_final = np.zeros(n_classes)

        w_avg = np.zeros_like(w)
        w_trans_avg = np.zeros_like(w_trans)
        w_init_avg = np.zeros_like(w_init)
        w_final_avg = np.zeros_like(w_final)

        return cls(w, w_trans, w_init, w_final,
                   w_avg, w_trans_avg, w_init_avg, w_final_avg)

    def averages(self, k):
        coef_ = self.w_avg / k
        coef_init_ = self.w_init_avg / k
        coef_trans_ = self.w_trans_avg / k
        coef_final_ = self.w_final_avg / k
        return coef_, coef_init_, coef_trans_, coef_final_


class StructuredPerceptron(BaseSequenceClassifier):
    """Structured perceptron for sequence classification.

    This implements the averaged structured perceptron algorithm of Collins,
    with the addition of a learning rate.

    Parameters
    ----------
    decode : string, optional
        Decoding algorithm, either "bestfirst" or "viterbi" (default).

    learning_rate : float, optional
        Learning rate.

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
    def __init__(self, decode="viterbi", learning_rate=.1, max_iter=10,
                 random_state=None, verbose=0):
        self.decode = decode
        self.learning_rate = learning_rate
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

        state = EpochState.initial(n_classes, n_features)
        (w, w_trans, w_init, w_final,
         w_avg, w_trans_avg, w_init_avg, w_final_avg) = state

        lr = self.learning_rate
        rng = check_random_state(self.random_state)

        for it in xrange(1, self.max_iter + 1):
            if self.verbose:
                print("Iteration ", it, end="... ")
                sys.stdout.flush()

            sequence_ids = rng.permutation(lengths.shape[0])
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

                    Y_diff = Y_pred - Y_t_i
                    Y_diff *= lr
                    w -= safe_sparse_dot(Y_diff.T, X_i)

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


class OneEpochPerceptron(BaseSequenceClassifier):
    def __init__(self, decode='viterbi', learning_rate=.1, random_state=None):
        self.decode = decode
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, X, y, lengths, class_range):
        """Remember a set of training sequences.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Feature matrix of individual samples.

        y : array-like, shape (n_samples,)
            Target labels.

        lengths : array-like of integers, shape (n_sequences,)
            Lengths of the individual sequences in X, y. The sum of these
            should be n_samples.

        class_range : array-like of integers

        Returns
        -------
        self : OneEpochPerceptron
        """
        self.X = atleast2d_or_csr(X)
        self.y = y
        self.lengths = np.asarray(lengths)

        self.class_range = class_range
        self.Y_true = y.reshape(-1, 1) == class_range

        self.rng = check_random_state(self.random_state)
        self.end = np.cumsum(lengths)
        self.start = self.end - lengths

    def transform(self, state):
        """Compute updated weights: do a single iteration over
        data and return (new_state, sum_loss) tuple.

        Parameters
        ----------
        state : EpochState
            Weighted mixture of parameters from all shards from
            previous epoch.

        Returns
        -------
        (new_state, sum_loss) tuple.

        """
        decode = self._get_decoder()

        lr = self.learning_rate
        Y_true = self.Y_true
        n_classes = Y_true.shape[1]

        rng = check_random_state(self.random_state)
        sequence_ids = rng.permutation(self.lengths.shape[0])

        sum_loss = 0
        s = state.copy()

        for i in sequence_ids:
            X_i = self.X[self.start[i]:self.end[i]]
            Score = safe_sparse_dot(X_i, s.w.T)
            y_pred = decode(Score, s.w_trans, s.w_init, s.w_final)
            y_t_i = self.y[self.start[i]:self.end[i]]
            loss = (y_pred != y_t_i).sum()

            if loss:
                sum_loss += loss

                Y_t_i = Y_true[self.start[i]:self.end[i]]
                Y_pred = y_pred.reshape(-1, 1) == self.class_range
                Y_pred = Y_pred.astype(np.float64)

                Y_diff = Y_pred - Y_t_i
                Y_diff *= lr
                s.w[:] -= safe_sparse_dot(Y_diff.T, X_i)

                t_trans = count_trans(y_t_i, n_classes)
                p_trans = count_trans(y_pred, n_classes)
                s.w_trans[:] -= lr * (p_trans - t_trans)
                s.w_init[:] -= lr * (Y_pred[0] - Y_true[self.start[i]])
                s.w_final[:] -= lr * (Y_pred[-1] - Y_true[self.end[i] - 1])

            s.w_avg[:] += s.w
            s.w_trans_avg[:] += s.w_trans
            s.w_init_avg[:] += s.w_init
            s.w_final_avg[:] += s.w_final

        return s, sum_loss

    @classmethod
    def split_task(cls, X, y, lengths, n_jobs, **init_kwargs):
        """
        Return a list of OneEpochPerceptron instances.
        """
        X = atleast2d_or_csr(X)
        lengths = np.asarray(lengths)

        classes = np.unique(y)
        class_range = np.arange(len(classes))

        perceptrons = []
        for shard, length_shard in SequenceShards(lengths, n_jobs):
            perc = OneEpochPerceptron(**init_kwargs)
            perc.fit(X[shard], y[shard], lengths[length_shard], class_range)
            perceptrons.append(perc)

        return perceptrons


def mixed_epoch_states(transform_results):
    k = len(transform_results)
    states = [state for state, sum_loss in transform_results]
    state = EpochState(*[sum(coeffs) / k for coeffs in zip(*states)])
    sum_loss = sum(loss for state, loss in transform_results)
    return state, sum_loss


class ParallelStructuredPerceptron(BaseSequenceClassifier):
    """Structured perceptron for sequence classification.

    Similar to StructuredPerceptron, but uses iterative parameter mixing
    to enable parallel learning.

    XXX: "parallel" doesn't work yet.

    References
    ----------
    Ryan Mcdonald, Keith Hall, and Gideon Mann (2010)
    Distributed training strategies for the structured perceptron. NAACL'10.
    """
    def __init__(self, decode="viterbi", learning_rate=.1, max_iter=10,
                 random_state=None, verbose=0, n_jobs=1):
        self.decode = decode
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs

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
        self : ParallelStructuredPerceptron
        """

        classes, y = np.unique(y, return_inverse=True)
        class_range = np.arange(len(classes))
        Y_true = y.reshape(-1, 1) == class_range
        n_classes = Y_true.shape[1]

        X = atleast2d_or_csr(X)
        n_samples, n_features = X.shape

        rng = check_random_state(self.random_state)

        perceptrons = OneEpochPerceptron.split_task(X, y, lengths, self.n_jobs,
                                                    decode=self.decode,
                                                    learning_rate=self.learning_rate,
                                                    random_state=rng)

        state = EpochState.initial(n_classes, n_features)
        for it in range(1, self.max_iter + 1):
            if self.verbose:
                print("Iteration ", it, end="... ")
                sys.stdout.flush()

            # XXX: how to make this parallel without copying X, y, etc.
            # on each iteration?
            results = [p.transform(state) for p in perceptrons]

            #with futures.ThreadPoolExecutor(self.n_jobs) as executor:
            #    jobs = [executor.submit(p.transform, state) for p in perceptrons]
            #    results = [f.result() for f in futures.as_completed(jobs)]

            state, sum_loss = mixed_epoch_states(results)

            if self.verbose:
                # XXX the loss reported is that for w, but the one for
                # w_avg is what matters for early stopping.
                print("Loss = {0:.4f}".format(sum_loss / n_samples))

        coefs = state.averages(it*len(lengths))
        self.coef_, self.coef_init_, self.coef_trans_, self.coef_final_ = coefs
        self.classes_ = classes

        return self
