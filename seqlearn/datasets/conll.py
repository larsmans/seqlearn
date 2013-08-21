# Copyright 2013 Lars Buitinck

from contextlib import closing
from itertools import chain, groupby, imap

import numpy as np
from sklearn.feature_extraction import FeatureHasher


def load_conll(f, features, n_features=(2 ** 16), split=False):
    """Load CoNLL file, extract features on the tokens and hash them.

    Parameters
    ----------
    f : {string, file-like}
        Input file.
    features : callable
        Feature extraction function. Must take a list of tokens (see below)
        and an index into this list.
    n_features : integer, optional
        Number of columns in the output.
    split : boolean, default=False
        Whether to split lines on whitespace beyond what is needed to parse
        out the labels. This is useful for CoNLL files that have extra columns
        containing information like part of speech tags.
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
    lines = imap(str.strip, f)
    groups = (grp for nonempty, grp in groupby(lines, bool) if nonempty)

    for group in groups:
        group = list(group)

        obs, lbl = zip(*(ln.rsplit(None, 1) for ln in group))
        if split:
            obs = [x.split() for x in obs]

        labels.extend(lbl)
        lengths.append(len(lbl))
        for i in xrange(len(obs)):
            yield features(obs, i)


def _open(f):
    return closing(open(f) if isinstance(f, basestring) else f)
