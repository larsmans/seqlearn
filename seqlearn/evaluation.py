from functools import partial
from warnings import warn

import numpy as np

from ._utils import check_random_state


def bio_f_score(y_true, y_pred):
    """F-score for BIO-tagging scheme, as used by CoNLL.

    This F-score variant is used for evaluating named-entity recognition and
    related problems, where the goal is to predict segments of interest within
    sequences and mark these as a "B" (begin) tag followed by zero or more "I"
    (inside) tags. A true positive is then defined as a BI* segment in both
    y_true and y_pred, with false positives and false negatives defined
    similarly.

    Support for tags schemes with classes (e.g. "B-NP") are limited: reported
    scores may be too high for inconsistent labelings.

    Parameters
    ----------
    y_true : array-like of strings, shape (n_samples,)
        Ground truth labeling.

    y_pred : array-like of strings, shape (n_samples,)
        Sequence classifier's predictions.

    Returns
    -------
    f : float
        F-score.
    """

    if len(y_true) != len(y_pred):
        msg = "Sequences not of the same length ({} != {})."""
        raise ValueError(msg.format(len(y_true), len(y_pred)))

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    is_b = partial(np.char.startswith, prefix="B")

    where = np.where
    t_starts = where(is_b(y_true))[0]
    p_starts = where(is_b(y_pred))[0]

    # These lengths are off-by-one because we skip the "B", but that's ok.
    # http://stackoverflow.com/q/17929499/166749
    t_lengths = np.diff(where(is_b(np.r_[y_true[y_true != 'O'], ['B']]))[0])
    p_lengths = np.diff(where(is_b(np.r_[y_pred[y_pred != 'O'], ['B']]))[0])

    t_segments = set(zip(t_starts, t_lengths, y_true[t_starts]))
    p_segments = set(zip(p_starts, p_lengths, y_pred[p_starts]))

    # tp = len(t_segments & p_segments)
    # fn = len(t_segments - p_segments)
    # fp = len(p_segments - t_segments)
    tp = sum(x in t_segments for x in p_segments)
    fn = sum(x not in p_segments for x in t_segments)
    fp = sum(x not in t_segments for x in p_segments)

    if tp == 0:
        # special-cased like this in CoNLL evaluation
        return 0.

    precision = tp / float(tp + fp)
    recall = tp / float(tp + fn)

    return 2. * precision * recall / (precision + recall)


class SequenceKFold(object):
    """Sequence-aware (repeated) k-fold CV splitter.

    Uses a greedy heuristic to partition input sequences into sets with roughly
    equal numbers of samples, while keeping the sequences intact.

    Parameters
    ----------
    lengths : array-like of integers, shape (n_samples,)
        Lengths of sequences, in the order in which they appear in the dataset.

    n_folds : int, optional
        Number of folds.

    n_iter : int, optional
        Number of iterations of repeated k-fold splitting. The default value
        is 1, meaning a single k-fold split; values >1 give repeated k-fold
        with shuffling (see below).

    shuffle : boolean, optional
        Whether to shuffle sequences.

    random_state : {np.random.RandomState, integer}, optional
        Random state/random seed for shuffling.
    """

    def __init__(self, lengths, n_folds=3, n_iter=1, shuffle=False,
                 random_state=None):
        if n_iter > 1 and not shuffle:
            warn("n_iter > 1 makes little sense without shuffling!")
        self.lengths = lengths
        self.n_folds = n_folds
        self.n_iter = n_iter
        self.random_state = random_state
        self.shuffle = shuffle

    def __iter__(self):
        rng = check_random_state(self.random_state)
        lengths = self.lengths
        starts = np.cumsum(lengths) - lengths
        n_samples = np.sum(lengths)

        seq_ind = np.arange(len(lengths))

        for _ in xrange(self.n_iter):
            if self.shuffle:
                rng.shuffle(seq_ind)

            folds = [[] for _ in range(self.n_folds)]
            samples_per_fold = np.zeros(self.n_folds, dtype=int)

            # Greedy strategy: always append to the currently smallest fold
            for i in seq_ind:
                seq = (starts[i], starts[i] + lengths[i])
                fold_idx = np.argmin(samples_per_fold)
                folds[fold_idx].append(seq)
                samples_per_fold[fold_idx] += lengths[i]

            for f in folds:
                test = np.zeros(n_samples, dtype=bool)
                for start, end in f:
                    test[start:end] = True

                train = ~test
                train = np.where(train)[0]
                test = np.where(test)[0]
                yield train, test

    def __len__(self):
        return self.n_folds * self.n_iter
