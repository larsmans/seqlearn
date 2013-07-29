import numpy as np


def bio_f_score(y_true, y_pred):
    """F-score for BIO-tagging scheme, as used by CoNLL.

    This F-score variant is used for evaluating named-entity recognition and
    related problems, where the goal is to predict segments of interest within
    sequences and mark these as a "B" (begin) tag followed by zero or more "I"
    (inside) tags. A true positive is then defined as a BI* segment in both
    y_true and y_pred, with false positives and false negatives defined
    similarly.

    Tags schemes with classes (e.g. "B-NP") are not supported at the moment.

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

    where = np.where
    t_starts = where(y_true == "B")[0]
    p_starts = where(y_pred == "B")[0]

    # These lengths are off-by-one because we skip the "B", but that's ok.
    # http://stackoverflow.com/q/17929499/166749
    t_lengths = np.diff(where(np.r_[y_true[y_true != 'O'], ['B']] == 'B')[0])
    p_lengths = np.diff(where(np.r_[y_pred[y_pred != 'O'], ['B']] == 'B')[0])

    t_segments = set(zip(t_starts, t_lengths))
    p_segments = set(zip(p_starts, p_lengths))

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
