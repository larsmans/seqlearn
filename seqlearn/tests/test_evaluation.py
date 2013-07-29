from nose.tools import assert_equal

from seqlearn.evaluation import bio_f_score


def test_bio_f_score():
    # Outputs from with the "conlleval" Perl script from CoNLL 2002.
    examples = [
        ("OBIO", "OBIO", 1.),
        ("BII", "OBI", 0.),
        ("BB", "BI", 0.),
        ("BBII", "BBBB", 1 / 3.),
        ("BOOBIB", "BOBOOB", 2 / 3.),
    ]

    for y_true, y_pred, score in examples:
        y_true = list(y_true)
        y_pred = list(y_pred)
        assert_equal(score, bio_f_score(y_true, y_pred))
