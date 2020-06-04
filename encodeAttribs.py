from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X[self.attribute_names]
    
    
class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X,y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X], index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)

def preprocess_pipeline(num_attribs, degree, cat_attribs):
	return FeatureUnion(transformer_list=[("num_pipeline", num_pipeline(num_attribs, degree)),("cat_pipeline", cat_pipeline(cat_attribs))])


def num_pipeline(num_attribs, deg):
	return Pipeline([("select_numeric", DataFrameSelector(num_attribs)),('scaler', MinMaxScaler()),('poly', PolynomialFeatures(degree=deg)),("imputer", SimpleImputer(strategy="median")),])

def cat_pipeline(cat_attribs):
	return Pipeline([("select_cat", DataFrameSelector(cat_attribs)),("imputer", MostFrequentImputer()),("cat_encoder", OneHotEncoder(sparse=False))])
