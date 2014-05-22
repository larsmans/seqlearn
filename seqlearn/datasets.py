# Copyright 2013 Lars Buitinck

from contextlib import closing
from itertools import chain, groupby

import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.externals import six


def load_conll(f, features, n_features=(2 ** 16), split=False):
    """Load CoNLL file, extract features on the tokens and vectorize them.

    The ConLL file format is a line-oriented text format that describes
    sequences in a space-separated format, separating the sequences with
    blank lines. Typically, the last space-separated part is a label.

    Since the tab-separated parts are usually tokens (and maybe things like
    part-of-speech tags) rather than feature vectors, a function must be
    supplied that does the actual feature extraction. This function has access
    to the entire sequence, so that it can extract context features.

    A ``sklearn.feature_extraction.FeatureHasher`` (the "hashing trick")
    is used to map symbolic input feature names to columns, so this function
    dos not remember the actual input feature names.

    Parameters
    ----------
    f : {string, file-like}
        Input file.
    features : callable
        Feature extraction function. Must take a list of tokens l that
        represent a single sequence and an index i into this list, and must
        return an iterator over strings that represent the features of l[i].
    n_features : integer, optional
        Number of columns in the output.
    split : boolean, default=False
        Whether to split lines on whitespace beyond what is needed to parse
        out the labels. This is useful for CoNLL files that have extra columns
        containing information like part of speech tags.

    Returns
    -------
    X : scipy.sparse matrix, shape (n_samples, n_features)
        Samples (feature vectors), as a single sparse matrix.
    y : np.ndarray, dtype np.string, shape n_samples
        Per-sample labels.
    lengths : np.ndarray, dtype np.int32, shape n_sequences
        Lengths of sequences within (X, y). The sum of these is equal to
        n_samples.
    """
    fh = FeatureHasher(n_features=n_features, input_type="string")
    labels = []
    lengths = []

    with _open(f) as f:
        raw_X = _conll_sequences(f, features, labels, lengths, split)
        X = fh.transform(raw_X)

    return X, np.asarray(labels), np.asarray(lengths, dtype=np.int32)


def _conll_sequences(f, features, labels, lengths, split):
    # Divide input into blocks of empty and non-empty lines.
    lines = (str.strip(line) for line in  f)
    groups = (grp for nonempty, grp in groupby(lines, bool) if nonempty)

    for group in groups:
        group = list(group)

        obs, lbl = zip(*(ln.rsplit(None, 1) for ln in group))
        if split:
            obs = [x.split() for x in obs]

        labels.extend(lbl)
        lengths.append(len(lbl))
        for i in six.moves.xrange(len(obs)):
            yield features(obs, i)


def _open(f):
    return closing(open(f) if isinstance(f, six.string_types) else f)
