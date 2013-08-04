from numpy.testing import assert_array_equal

import numpy as np

from seqlearn._utils import count_trans


def test_count_trans():
    y = np.asarray([1, 1, 2, 1, 1, 2, 0, 1, 0])
    expected = [[0, 1, 0],
                [1, 2, 2],
                [1, 1, 0]]
    assert_array_equal(count_trans(y, 3), expected)

    assert_array_equal(count_trans(y, 4),
                       np.c_[np.r_[expected, np.zeros((1, 3))], np.zeros(4)])
