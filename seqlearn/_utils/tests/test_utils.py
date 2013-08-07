from nose.tools import assert_raises
from numpy.testing import assert_array_equal

import numpy as np

from seqlearn._utils import count_trans, validate_lengths


def test_count_trans():
    y = np.asarray([1, 1, 2, 1, 1, 2, 0, 1, 0])
    expected = [[0, 1, 0],
                [1, 2, 2],
                [1, 1, 0]]
    assert_array_equal(count_trans(y, 3), expected)

    assert_array_equal(count_trans(y, 4),
                       np.c_[np.r_[expected, np.zeros((1, 3))], np.zeros(4)])


def test_count_trans_dtype():
    y = np.asarray([1, 1, 2, 1, 1, 2, 0, 1, 0], dtype=np.int8)
    # Strangely, Cython raises a ValueError instead of a TypeError.
    assert_raises(ValueError, count_trans, y, 3)


def test_validate_lengths():
    start, end = validate_lengths(50, [4, 5, 41])
    assert_array_equal(start, [0, 4, 9])
    assert_array_equal(end, [4, 9, 50])

    assert_raises(ValueError, validate_lengths, 5, [4, 2])
