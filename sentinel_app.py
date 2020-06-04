import streamlit as st
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from encodeAttribs import preprocess_pipeline

log_clf = pickle.load(open('logistic_clf.sav','rb'))
gnb_clf = pickle.load(open('gnb_clf.sav','rb'))
pipe = pickle.load(open('transform.sav', 'rb'))

#do pipe.transform(data) to be able to perform predictions on it