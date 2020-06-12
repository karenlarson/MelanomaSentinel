import pandas as pd
from cleanData import cleanData
from sklearn.model_selection import train_test_split
from encodeAttribs import preprocess_pipeline
from baseLine import BaseLine
from sklearn.model_selection import cross_val_score
import pickle 
from sklearn.linear_model import LogisticRegression
import pandas as pd
import psycopg2 as sql
from classifierEvaluation import classifierEvaluation


"""
This is the main function for creating the melanoma prediction model. It calls functions to
1) Get the data from a postgresql database
2) calls cleanData to clean data (see cleanData.py for the description of how that's done)
3) Splits the data into train/CV/test sets
4) Calls preprocess_pipeline in encodeAttribs.py to create numerical features of degree 3
and one-hot encode the cateogrical attributes
5) Trains a baseline classifier (see BaseLine.py for details)
6) Fits a logistic regression classifier using sklearn
7) Validates the model using classifierEvaluation.py
8) Saves the model using pickle for the web-app's use
"""

#cleaning the data in cleanData
sql_mela = sql.connect(host="localhost", user="karenlarson", dbname="melanoma")
query = """SELECT "AGE", "SEX", "DEPTH", "ULCERATION", "MITOSES", "CS_EXTENSION", "PRIMARY_SITE", "CS_LYMPH_NODE_METS"
            FROM melanoma"""

#query for matching JCO data            
#query = """SELECT "AGE", "SEX", "DEPTH", "ULCERATION", "MITOSES", "CS_EXTENSION", "PRIMARY_SITE", "CS_LYMPH_NODE_METS"
#           FROM melanoma
#           WHERE ("DEPTH" <= 400 AND "DEPTH" >= 100)
#           OR (("DEPTH" < 100 AND "DEPTH" >= 75) AND
#           ("ULCERATION" >= 1 OR "MITOSES" >= 1 OR "AGE" < 40 )) OR 
#           (("DEPTH" < 75 AND "DEPTH" >= 54) AND 
#           (("MITOSES" >= 1 AND "ULCERATION" >= 1) OR ("ULCERATION" >= 1
#           AND "AGE" < 40) OR ("MITOSES" >= 1 AND "AGE" < 40)));"""

mela = pd.read_sql_query(query, sql_mela)
X, y = cleanData(mela)

#split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=20)
X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size=0.10, random_state=20)

#using functions so easily adjusted
num_attribs = ['AGE','DEPTH', "CS_EXTENSION", "MITOSES"]
cat_attribs = [ 'SEX', 'ULCERATION', 'PRIMARY_SITE']

pipe = preprocess_pipeline(num_attribs, 3, cat_attribs).fit(X_train)
#transforming features according to the pipeline
x_train = pipe.transform(X_train)
x_cv = pipe.transform(X_cv)
x_test = pipe.transform(X_test)

#getting baseline accuracy (although not best comparison in this case)
base_line_clf = BaseLine()
base_scores = cross_val_score(base_line_clf,x_train,y_train,cv=10,scoring="accuracy")
print(base_scores.mean())

#fitting to a logistic regression classifier
log_clf = LogisticRegression(solver="liblinear")
log_clf.fit(x_train,y_train)

# summarize feature importance
importance = log_clf.coef_[0]
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))

#evaluating the goodness of the logistic regression classifier
classifierEvaluation(log_clf, 0.005, x_train, y_train, "Logistic Regression", "Training Data")
classifierEvaluation(log_clf, 0.005, x_cv, y_cv, "Logistic Regression", "Validation Data")
classifierEvaluation(log_clf, 0.005, x_test, y_test, "Logistic Regression", "Test Data")

#dumps the trained model using pickle for the web-app to use.
filename = 'logistic_clf.sav'
pickle.dump(log_clf, open(filename, 'wb'))


