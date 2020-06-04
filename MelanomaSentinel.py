import pandas as pd
from cleanData import cleanData
from sklearn.model_selection import train_test_split
from encodeAttribs import preprocess_pipeline
from baseLine import BaseLine
from sklearn.model_selection import cross_val_score
from logReg import log_clf_test
from bayesReg import gnb_clf
import pickle 
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

#cleaning the data in cleanData
X, y = cleanData('Melanoma.csv')

#split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=20)

#using functions so easily adjusted
num_attribs = ['AGE', 'DEPTH', "MITOSES", "CS_EXTENSION"]
cat_attribs = ['SEX', 'RACE', 'SPANISH_HISPANIC_ORIGIN','ULCERATION', 'PRIMARY_SITE']

pipe = preprocess_pipeline(num_attribs, 3, cat_attribs).fit(X_train)
x_train = pipe.transform(X_train)

x_cv, x_train = x_train[:10000, :], x_train[10000:,:]
y_cv, y_train = y_train[:10000], y_train[10000:]

base_line_clf = BaseLine()
base_scores = cross_val_score(base_line_clf,x_train,y_train,cv=10,scoring="accuracy")
print(base_scores.mean())

log_clf = log_clf_test(x_train, x_cv, y_train, y_cv)
gnb = gnb_clf(x_train, x_cv, y_train, y_cv)

filename = 'logistic_clf.sav'
pickle.dump(log_clf, open(filename, 'wb'))
filename_gnb = 'gnb_clf.sav'
pickle.dump(gnb, open(filename_gnb, 'wb'))

filename = 'transform.sav'
pickle.dump(pipe, open(filename, 'wb'))


