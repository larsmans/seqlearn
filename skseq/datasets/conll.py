# Copyright 2013 Lars Buitinck

from contextlib import closing
from itertools import chain, groupby, imap

import numpy as np
from sklearn.feature_extraction import FeatureHasher


def load_conll(f, features):
    fh = FeatureHasher(input_type="string")
    labels = []

    with _open(f) as f:
        raw_X = _conll_sequences(f, features, labels)
        X = fh.transform(raw_X)

    return X, np.asarray(labels)


def _conll_sequences(f, features, labels, boundary="_OUT"):
    # Divide input blocks of empty and non-empty lines.
    # Make sure first and last blocks have empty lines.
    lines = chain([""], imap(str.strip, f), [""])
    groups = groupby(lines, bool)
    next(groups)

    for nonempty, group in groups:
        assert nonempty
        group = list(group)

        next(groups)    # consume empty lines

        obs, lbl = zip(*(ln.rsplit(None, 1) for ln in group))

        labels.append(boundary)
        yield [boundary]

        labels.extend(lbl)
        for i in xrange(len(obs)):
            yield features(obs, i)

        labels.append(boundary)
        yield [boundary]


def _open(f):
    return closing(open(f) if isinstance(f, basestring) else f)
