# Copyright 2013 Lars Buitinck

from contextlib import closing
from itertools import chain, groupby, imap

import numpy as np
from sklearn.feature_extraction import FeatureHasher


def load_conll(f, features):
    fh = FeatureHasher(input_type="string")
    labels = []
    offsets = []

    with _open(f) as f:
        raw_X = _conll_sequences(f, features, labels, offsets)
        X = fh.transform(raw_X)

    return X, np.asarray(labels), np.asarray(offsets, dtype=np.int32)


def _conll_sequences(f, features, labels, offsets):
    # Divide input blocks of empty and non-empty lines.
    # Make sure first and last blocks have empty lines.
    lines = chain([""], imap(str.strip, f), [""])
    groups = groupby(lines, bool)
    next(groups)

    offset = 0

    for nonempty, group in groups:
        assert nonempty
        group = list(group)

        next(groups)    # consume empty lines

        obs, lbl = zip(*(ln.rsplit(None, 1) for ln in group))

        labels.extend(lbl)
        offsets.append(offset)
        offset += len(lbl)
        for i in xrange(len(obs)):
            yield features(obs, i)


def _open(f):
    return closing(open(f) if isinstance(f, basestring) else f)
