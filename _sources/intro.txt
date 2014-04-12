Introduction
============

seqlearn extends the `scikit-learn <http://scikit-learn.org>`_
machine learning library to deal with sequence classification:
sequences of observations that must be individually labeled,
but where the order in which they appear matters.

seqlearn mimicks the basic scikit-learn ``fit``/``predict`` API
and tries to stay compatible with scikit-learn's data formats,
but adds an argument to the scikit-learn methods that encodes the structure
of the input. This argument is called ``lengths``
and should be an array of integers denoting the respective lengths
of sequences in ``(X, y)``.

For example, if ``X`` and ``y`` both have length (``shape[0]``) of 10, then
``lengths=[6, 4]`` encodes the information that ``(X[:6], y[:6])`` and
``(X[6:10], y[6:10])`` are both coherent sequences.
This encoding of sequence information allows for a fast implementation
using NumPy's vectorized operations.
