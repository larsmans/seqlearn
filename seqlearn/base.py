import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score

from ._decode import DECODERS
from ._utils import atleast2d_or_csr, safe_sparse_dot, validate_lengths


# XXX Should we even derive from ClassifierMixin here?
# We override all the methods.
class BaseSequenceClassifier(BaseEstimator, ClassifierMixin):
    def _get_decoder(self):
        try:
            return DECODERS[self.decode]
        except KeyError:
            return ValueError("Unknown decoder {0!r}".format(self.decode))

    def predict(self, X, lengths=None):
        """Predict labels/tags for samples X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Feature matrix.

        lengths : array-like of integer, shape (n_sequences,), optional
            Lengths of sequences in X. If not given, X is assumed to be a
            single sequence of length n_samples.

        Returns
        -------
        y : array, shape (n_samples,)
            Labels per sample in X.
        """
        X = atleast2d_or_csr(X, dtype=np.float64)
        Scores = safe_sparse_dot(X, self.coef_.T)
        decode = DECODERS[self.decode]

        if lengths is None:
            y = decode(Scores, self.coef_trans_,
                       self.coef_init_, self.coef_final_)
        else:
            start, end = validate_lengths(X.shape[0], lengths)

            y = [decode(Scores[start[i]:end[i]], self.coef_trans_,
                        self.coef_init_, self.coef_final_)
                 for i in xrange(len(lengths))]
            y = np.hstack(y)

        return self.classes_[y]


    def score(self, X, y, lengths=None):
        """Returns the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.

        y : array-like, shape = (n_samples,)
            True labels for X.

        lengths : array-like of integer, shape (n_sequences,), optional
            Lengths of sequences in X. If not given, X is assumed to be a
            single sequence of length n_samples.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X, lengths) wrt. y.
        """
        return accuracy_score(y, self.predict(X, lengths))
