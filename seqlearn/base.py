import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.externals import six

from ._decode import DECODERS
from ._utils import atleast2d_or_csr, safe_sparse_dot, validate_lengths


# XXX Should we even derive from ClassifierMixin here?
# We override all the methods.
class BaseSequenceClassifier(BaseEstimator, ClassifierMixin):
    def _get_decoder(self):
        try:
            return DECODERS[self.decode]
        except KeyError:
            raise ValueError("Unknown decoder {0!r}".format(self.decode))

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
        X = atleast2d_or_csr(X)
        scores = safe_sparse_dot(X, self.coef_.T)
        if hasattr(self, "coef_trans_"):
            n_classes = len(self.classes_)
            coef_t = self.coef_trans_.T.reshape(-1, self.coef_trans_.shape[-1])
            trans_scores = safe_sparse_dot(X, coef_t.T)
            trans_scores = trans_scores.reshape(-1, n_classes, n_classes)
        else:
            trans_scores = None

        decode = self._get_decoder()

        if lengths is None:
            y = decode(scores, trans_scores, self.intercept_trans_,
                       self.intercept_init_, self.intercept_final_)
        else:
            start, end = validate_lengths(X.shape[0], lengths)

            y = [decode(scores[start[i]:end[i]], trans_scores,
                        self.intercept_trans_, self.intercept_init_,
                        self.intercept_final_)
                 for i in six.moves.xrange(len(lengths))]
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
