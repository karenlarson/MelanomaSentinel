from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

def gnb_clf(x_train, x_cv, y_train, y_cv):
	gnb = GaussianNB()
	y_pred = gnb.fit(x_train, y_train).predict(x_train)
	gnb_score = accuracy_score(y_train, y_pred)
	#print(y_test)
	gnb_f1 = f1_score(y_train,y_pred)
	gnb_recall = recall_score(y_train, y_pred)
	gnb_precision = precision_score(y_train,y_pred)
	print(f"For Naive Bayes, the training accuracy is {gnb_score}, the precision is {gnb_precision}, the recall is {gnb_recall}, and the f1 score is {gnb_f1}")

	y_pred = gnb.fit(x_train, y_train).predict(x_cv)
	gnb_score = accuracy_score(y_cv, y_pred)
	gnb_f1 = f1_score(y_cv,y_pred)
	gnb_recall = recall_score(y_cv, y_pred)
	gnb_precision = precision_score(y_cv,y_pred)
	print(f"For Naive Bayes, the validation accuracy is {gnb_score}, the precision is {gnb_precision}, the recall is {gnb_recall}, and the f1 score is {gnb_f1}")
	#print(confusion_matrix(y_cv, y_pred))

	return gnb
