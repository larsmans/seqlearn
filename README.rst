.. -*- mode: rst -*-

seqlearn
========

seqlearn is a sequence classification toolkit for Python. It is designed to
extend `scikit-learn <http://scikit-learn.org>`_ and offer as similar as
possible an API.


Compiling and installing
------------------------

Get NumPy >=1.6, SciPy >=0.11, Cython >=0.20.2 and a recent version of
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

Check how well you did on a validation set, say ``validation.txt``::

    >>> X_test, y_test, lengths_test = load_conll("validation.txt", features)
    >>> from seqlearn.evaluation import bio_f_score
    >>> y_pred = clf.predict(X_test, lengths_test)
    >>> print(bio_f_score(y_test, y_pred))

For more information, see the `documentation
<http://larsmans.github.io/seqlearn>`_.


|Travis|_

.. |Travis| image:: https://api.travis-ci.org/larsmans/seqlearn.png?branch=master
.. _Travis: https://travis-ci.org/larsmans/seqlearn
