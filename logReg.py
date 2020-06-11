from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from classifierEvaluation import classifierEvaluation

def log_clf_test(x_train, x_cv, y_train, y_cv):
	log_clf = LogisticRegression(solver="liblinear")
	log_clf.fit(x_train,y_train)
	log_scores = cross_val_score(log_clf, x_train,y_train,cv=10,scoring="accuracy")
	#y_pred = log_clf.predict(x_train)
	#log_score = accuracy_score(y_train,y_pred)

	importance = log_clf.coef_[0]
	# summarize feature importance
	for i,v in enumerate(importance):
		print('Feature: %0d, Score: %.5f' % (i,v))

	classifierEvaluation(log_clf, 0.005, x_train, y_train, "Logistic Regression", "Training Data")

	classifierEvaluation(log_clf, 0.005, x_cv, y_cv, "Logistic Regression", "Validation Data")
	#print(confusion_matrix(y_train, y_pred))
	#log_f1 = f1_score(y_train,y_pred)
	#log_recall = recall_score(y_train, y_pred)
	#log_precision = precision_score(y_train,y_pred)
	#print(f"For the training data, the accuracy is {log_scores.mean()}, precision is {log_precision}, the recall is {log_recall}, and the f1 score is {log_f1}")


	#dealing with the cross validation data now
	#y_pred = log_clf.predict(x_cv)
	#log_score = accuracy_score(y_cv, y_pred)
	#log_f1 = f1_score(y_cv,y_pred)
	#log_recall = recall_score(y_cv, y_pred)
	#log_precision = precision_score(y_cv,y_pred)
	#print(f"For the validation data, the accuracy is {log_score}, the precision is {log_precision}, the recall is {log_recall}, and the f1 score is {log_f1}")



	return log_clf
