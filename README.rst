.. -*- mode: rst -*-

seqlearn
========

seqlearn is a sequence classification toolkit for Python. It is designed to
extend `scikit-learn <http://scikit-learn.org>`_ and offer as similar as
possible an API.


Compiling and installing
------------------------

Get NumPy >=1.4, SciPy >=0.10, Cython >=0.19 and a recent version of
scikit-learn. Then issue::

    python setup.py install

to install seqlearn.

If you want to use seqlearn from its source directory without installing,
you have to compile first::

    python setup.py build_ext --inplace


Getting started
---------------

The easiest way to start using seqlearn is to fetch a dataset in CoNLL 2000
format. Define a task-specific feature extraction function, e.g.::

    >>> def features(sequence, i):
    ...     yield "word=" + sequence[i].lower()
    ...     if sequence[i].isupper():
    ...         yield "Uppercase"
    ...

Load the training file, say ``train.txt``::

    >>> from seqlearn.datasets import load_conll
    >>> X_train, y_train, lengths_train = load_conll("train.txt", features)

Train a model::

    >>> from seqlearn.perceptron import StructuredPerceptron
    >>> clf = StructuredPerceptron()
    >>> clf.fit(X_train, y_train, lengths_train)

Check how well you did on a validation set, say ``validation.txt``:

    >>> X_test, y_test, lengths_test = load_conll("validation.txt", features)
    >>> from seqlearn.evaluation import bio_f_score
    >>> y_pred = clf.predict(X_test, lengths_test)
    >>> print(bio_f_score(y_test, y_pred))


API
---

seqlearn tries to mimick the scikit-learn classifier API and stay compatible
with scikit-learn's data formats, but those are designed to represent sets of
unrelated samples (as rows in a matrix ``X`` and elements of a vector ``y``).
Therefore, the information about which samples belong to which sequences must
be encoded separately. For this purpose, seqlearn methods accept an array
called ``lengths`` which contains the lengths of the sequences in ``(X, y)``.

For example, if ``X`` and ``y`` both have length (``shape[0]``) of 10, then
``lengths=[6, 4]`` encodes the information that ``(X[:6], y[:6])`` and
``(X[6:10], y[6:10])`` are both coherent sequences.

This encoding of sequence information may seem cumbersome at first, but allows
for a fast implementation using NumPy's vectorized operations.
