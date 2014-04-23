from nose.tools import assert_equal, assert_true
from numpy.testing import assert_array_equal

import numpy as np
import re

from seqlearn.evaluation import (bio_f_score, SequenceKFold,
                                 whole_sequence_accuracy)


def test_bio_f_score():
    # Check against putputs from the "conlleval" Perl script from CoNLL 2002.
    examples = [
        ("OBIO", "OBIO", 1.),
        ("BII", "OBI", 0.),
        ("BB", "BI", 0.),
        ("BBII", "BBBB", 1 / 3.),
        ("BOOBIB", "BOBOOB", 2 / 3.),
        ("B-PER I-PER O B-PER I-PER O O B-LOC O".split(),
         "B-LOC I-LOC O B-PER I-PER O O B-LOC I-LOC".split(),
         1 / 3.)
    ]

    for y_true, y_pred, score in examples:
        y_true = list(y_true)
        y_pred = list(y_pred)
        assert_equal(score, bio_f_score(y_true, y_pred))


def test_accuracy():
    y_true = ["0111001", "1001", "00011111", "010101011", "1110"]
    y_pred = ["0010010", "1001", "00011110", "010101011", "1110"]
    assert_equal(.6, whole_sequence_accuracy(''.join(y_true), ''.join(y_pred),
                                             [len(y) for y in y_true]))


def test_kfold():
    sequences = [
        "BIIOOOBOO",
        "OOBIOOOBI",
        "OOOOO",
        "BIIOOOOO",
        "OBIOOBIIIII",
        "OOBII",
        "BIBIBIO",
    ]

    y = np.asarray(list(''.join(sequences)))

    for random_state in [75, 82, 91, 57, 291]:
        kfold = SequenceKFold(
            [len(s) for s in sequences],
            n_folds=3, shuffle=True, random_state=random_state
        )
        folds = list(iter(kfold))
        for train, lengths_train, test, lengths_test in folds:
            assert_true(np.issubdtype(train.dtype, np.integer))
            assert_true(np.issubdtype(test.dtype, np.integer))

            assert_true(np.all(train < len(y)))
            assert_true(np.all(test < len(y)))

            assert_equal(len(train), sum(lengths_train))
            assert_equal(len(test), sum(lengths_test))

            y_train = ''.join(y[train])
            y_test = ''.join(y[test])
            # consistent BIO labeling preserved
            assert_true(re.match(r'O*(?:BI*)O*', y_train))
            assert_true(re.match(r'O*(?:BI*)O*', y_test))


def test_kfold_repeated():
    lengths = [4, 5, 3, 7, 2, 3, 2, 5, 3, 7, 2, 3, 2, 4]
    n_samples = sum(lengths)

    # TODO test for warning when shuffle=False

    for n_folds in [2, 3, 4]:
        for n_iter in [2, 3, 4]:
            kfold = SequenceKFold(lengths, n_folds=n_folds, n_iter=n_iter,
                                  shuffle=True, random_state=42)

            assert_equal(len(kfold), n_folds * n_iter)

            folds = list(kfold)
            assert_equal(len(folds), n_folds * n_iter)

            for train, lengths_train, test, lengths_test in folds:
                assert_equal(sum(lengths_train), len(train))
                assert_equal(sum(lengths_test), len(test))
                assert_equal(len(train) + len(test), n_samples)
