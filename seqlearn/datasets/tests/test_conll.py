from nose.tools import assert_equal, assert_less, assert_true

from StringIO import StringIO

import scipy.sparse as sp

from seqlearn.datasets import load_conll


TEST_FILE = """

The Det
cat N
is      V
on Pre
the Det
 mat N
. Punc


Really Adv
. Punc

"""


def features(words, i):
    assert_true(isinstance(i, int))
    assert_less(-1, i)
    assert_less(i, len(words))

    yield words[i].lower()


def test_load_conll():
    n_nonempty = sum(1 for ln in TEST_FILE.splitlines() if ln.strip())

    X, y, seqid = load_conll(StringIO(TEST_FILE), features)
    assert_true(sp.isspmatrix(X))
    assert_equal(X.shape[0], n_nonempty)
    assert_equal(list(y),
                 ["Det", "N", "V", "Pre", "Det", "N", "Punc",
                  "Adv", "Punc"])
    assert_equal(list(seqid), [0] * 7 + [1] * 2)
