from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

#creating a selector to pick out the attributes from a given dataframe
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X[self.attribute_names]
#fills in missing values with the most frequent one
class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X,y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X], index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)

#main function that processes our numerical and categorical features to feed into the model
def preprocess_pipeline(num_attribs, degree, cat_attribs):
	return FeatureUnion(transformer_list=[("num_pipeline", num_pipeline(num_attribs, degree)),("cat_pipeline", cat_pipeline(cat_attribs))])

#selects the numeric features, creates polynomial features of a specificed degree, uses a median imputer, and mimaxscaler
def num_pipeline(num_attribs, deg):
	return Pipeline([("select_numeric", DataFrameSelector(num_attribs)),('scaler', StandardScaler()), ('poly', PolynomialFeatures(degree=deg)), ("imputer", SimpleImputer(strategy="median"))])
 
#selects categorical attributes, uses a most frequent imputer defined above, and one-hot encodes the attributes because we don't want a sense of similarity between our categories
def cat_pipeline(cat_attribs):
	return Pipeline([("select_cat", DataFrameSelector(cat_attribs)),("imputer", MostFrequentImputer()),("cat_encoder", OneHotEncoder(sparse=False))])
