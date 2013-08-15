from nose.tools import assert_raises
from numpy.testing import assert_array_equal, assert_array_almost_equal

import numpy as np

from seqlearn.hmm import MultinomialHMM


text = [w.split() for w in ["this DT",
                            "is V",
                            "a DT",
                            "test N",
                            "for IN",
                            "a DT",
                            "hidden Adj",
                            "Markov N",
                            "model N"]]
words, y = zip(*text)
lengths = [len(text)]

vocab, identities = np.unique(words, return_inverse=True)
X = (identities.reshape(-1, 1) == np.arange(len(vocab))).astype(int)


def test_hmm():
    n_features = X.shape[1]

    clf = MultinomialHMM()
    clf.fit(X, y, lengths)
    assert_array_equal(clf.classes_, ["Adj", "DT", "IN", "N", "V"])
    assert_array_equal(clf.predict(X), y)

    clf.set_params(decode="bestfirst")
    assert_array_equal(clf.predict(X), y)

    n_classes = len(clf.classes_)
    assert_array_almost_equal(np.ones(n_features),
                              np.exp(clf.coef_).sum(axis=0))
    assert_array_almost_equal(np.ones(n_classes),
                              np.exp(clf.intercept_trans_).sum(axis=0))
    assert_array_almost_equal(1., np.exp(clf.intercept_final_).sum())
    assert_array_almost_equal(1., np.exp(clf.intercept_init_).sum())


def test_hmm_validation():
    assert_raises(ValueError, MultinomialHMM(alpha=0).fit, X, y, lengths)
    assert_raises(ValueError, MultinomialHMM(alpha=-1).fit, X, y, lengths)
