from nose.tools import assert_equal, assert_less, assert_true
from numpy.testing import assert_array_equal
import scipy.sparse as sp

from sklearn.externals import six
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

    X, y, lengths = load_conll(six.moves.StringIO(TEST_FILE), features)
    assert_true(sp.isspmatrix(X))
    assert_equal(X.shape[0], n_nonempty)
    assert_equal(list(y),
                 ["Det", "N", "V", "Pre", "Det", "N", "Punc",
                  "Adv", "Punc"])
    assert_array_equal(lengths, [7, 2])


TEST_SPLIT = """
    foo ham O
    bar spam B
    baz eggs I

"""


def features_split(words, i):
    assert_true(isinstance(i, int))
    assert_less(-1, i)
    assert_less(i, len(words))

    x1, x2 = words[i]
    yield x1
    yield x2


def test_load_conll_split():
    X, y, _ = load_conll(six.moves.StringIO(TEST_SPLIT), features_split, split=True)
    assert_equal(list(y), list("OBI"))
