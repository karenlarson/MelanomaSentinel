import numpy as np
from sklearn.base import BaseEstimator

class BaseLine(BaseEstimator):
	"""Creates a Base-Line estimator by estimating all zeros, which is the majority class in this case"""
    def fit(self, X, y=None):
        pass
    def predict(self,X):
        return np.zeros((len(X),1))
