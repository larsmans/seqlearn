from sklearn.base import BaseEstimator, ClassifierMixin

from ._decode import DECODERS
from ._utils import safe_sparse_dot


class BaseSequenceClassifier(BaseEstimator, ClassifierMixin):
    def predict(self, X):
        Scores = safe_sparse_dot(X, self.coef_.T) + self.intercept_
        y = DECODERS[self.decode](Scores, self.coef_trans_,
                                  self.coef_init_, self.coef_final_)
        return self.classes_[y]
