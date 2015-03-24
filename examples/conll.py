# Copyright 2013 Lars Buitinck / University of Amsterdam.

"""
Generic sequence prediction script using CoNLL format.
"""

from __future__ import print_function
import fileinput
from glob import glob
import sys

from seqlearn.datasets import load_conll
from seqlearn.evaluation import bio_f_score
from seqlearn.perceptron import StructuredPerceptron
from sklearn.metrics import accuracy_score


def features(sentence, i):
    """Features for i'th token in sentence.

    Currently baseline named-entity recognition features, but these can
    easily be changed to do POS tagging or chunking.
    """

    word = sentence[i]

    yield "word:{}" + word.lower()

    if word[0].isupper():
        yield "CAP"

    if i > 0:
        yield "word-1:{}" + sentence[i - 1].lower()
        if i > 1:
            yield "word-2:{}" + sentence[i - 2].lower()
    if i + 1 < len(sentence):
        yield "word+1:{}" + sentence[i + 1].lower()
        if i + 2 < len(sentence):
            yield "word+2:{}" + sentence[i + 2].lower()


def describe(X, lengths):
    print("{0} sequences, {1} tokens.".format(len(lengths), X.shape[0]))


def load_data():
    files = glob('nerdata/*.bio')

    # 80% training, 20% test
    print("Loading training data...", end=" ")
    train_files = [f for i, f in enumerate(files) if i % 5 != 0]
    train = load_conll(fileinput.input(train_files), features)
    X_train, _, lengths_train = train
    describe(X_train, lengths_train)

    print("Loading test data...", end=" ")
    test_files = [f for i, f in enumerate(files) if i % 5 == 0]
    test = load_conll(fileinput.input(test_files), features)
    X_test, _, lengths_test = test
    describe(X_test, lengths_test)

    return train, test


if __name__ == "__main__":
    print(__doc__)

    #print("Loading training data...", end=" ")
    #X_train, y_train, lengths_train = load_conll(sys.argv[1], features)
    #describe(X_train, lengths_train)

    train, test = load_data()
    X_train, y_train, lengths_train = train
    X_test, y_test, lengths_test = test

    #print("Loading test data...", end=" ")
    #X_test, y_test, lengths_test = load_conll(sys.argv[2], features)
    #describe(X_test, lengths_test)

    clf = StructuredPerceptron(verbose=True, max_iter=10)
    print("Training %s" % clf)
    clf.fit(X_train, y_train, lengths_train)

    y_pred = clf.predict(X_test, lengths_test)
    print("Accuracy: %.3f" % (100 * accuracy_score(y_test, y_pred)))
    print("CoNLL F1: %.3f" % (100 * bio_f_score(y_test, y_pred)))
