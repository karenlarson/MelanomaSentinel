from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def classifierEvaluation(clf, threshold, X_data, y_data, classifier_name, data_descriptor):
	#creating predictions using classifier and a threshold for classifying as the positive class
	y_pred = np.where(clf.predict_proba(X_data)[:,1] > threshold, 1, 0)

	cf_mat = confusion_matrix(y_data, y_pred)
	tn, fp, fn, tp = cf_mat.ravel()
	#calculating the various measures of fit
	score = accuracy_score(y_data, y_pred)
	f1 = f1_score(y_data,y_pred)
	recall = recall_score(y_data, y_pred)
	precision = precision_score(y_data,y_pred)
	specificity = tn/(tn + fp)
	roc_auc = roc_auc_score(y_data, clf.predict_proba(X_data)[:,1])
	pos_predictive_value = tp/(tp + fp)
	neg_predictive_value = tn/(tn + fn)
	print(f"NEGATIVE PREDICTIVE VALUE {neg_predictive_value}")

	print("For the {} using {}, the accuracy is {:.4f}, the precision is {:.4f}, the recall is {:.4f}, the specificity is {:.4f}, the f1 score is {:.4f}, the auc is {:.4f}".format(data_descriptor, classifier_name, score, precision, recall, specificity, f1, roc_auc))


	#creating a seaborn visualization for the confusion matrix; splitting the colors based upon the Negative/Positive true labels.
	group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
	group_counts = ["{0:0.0f}".format(value) for value in cf_mat.flatten()]
	group_perct = [0]*4
	group_perct[0] = "{0:.2%}".format(tn/(tn + fp))
	group_perct[1] = "{0:.2%}".format(fp/(tn + fp))
	group_perct[2] = "{0:.2%}".format(fn/(fn + tp))
	group_perct[3] = "{0:.2%}".format(tp/(fn + tp))

	labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_perct)]
	labels = np.asarray(labels).reshape(2,2)
	#what we'll use for the heat coloring -- want the rows of the heat map to sum to 1 to get 
	#a sense of the specificty and recall of the method
	relative_percents = np.array([[tn/(tn + fp), fp/(tn + fp)],[fn/(fn + tp), tp/(fn + tp)]])
	print(cf_mat)
	if data_descriptor == "Training Data":
		colors = "Blues"
	elif data_descriptor == "Validation Data":
		colors = "Reds"
	else:
		colors = "Greens"
	sns.heatmap(relative_percents, annot=labels, annot_kws={"size":20}, fmt='', cmap = colors)
	plt.ylabel('True label', fontsize=18)
	plt.xlabel('Predicted label', fontsize=18)
	title = "Confusion Matrix for \n{} using {}".format(classifier_name, data_descriptor)
	plt.title(title, fontsize=18)
	plt.tick_params(axis='both', which='major', labelsize=15)

	plt.show()

def predictiveValue(clf, threshold, X_data, y_data, pipe):
	tmp = X_data
	tmp['TARGET'] = y_data

	c1 = tmp['DEPTH'] < 80
	c2 = tmp['ULCERATION'] == 0 

	T1a = tmp[c1 & c2]
	y1a = T1a['TARGET']
	T1a = T1a.drop(columns=["TARGET"])
	
	T1b = tmp[~c1 | (c1 & ~c2)]
	y1b = T1b['TARGET']
	T1b = T1b.drop(columns=["TARGET"])
	print(len(T1b))
	print(len(T1a))
	print(len(X_data))

	x_data = pipe.transform(X_data)
	t1a = pipe.transform(T1a)
	t1b = pipe.transform(T1b)


	y_pred = np.where(clf.predict_proba(t1b)[:,1] > threshold, 1, 0)

	cf_mat = confusion_matrix(y1b, y_pred)
	tn, fp, fn, tp = cf_mat.ravel()
	#calculating the various measures of fit
	neg_predictive_value = tn/(tn + fn)

	y_pred = np.where(clf.predict_proba(t1a)[:,1] > threshold, 1, 0)

	cf_mat = confusion_matrix(y1a, y_pred)
	tn, fp, fn, tp = cf_mat.ravel()


	pos_predictive_value = tp/(tp + fp)

	roc_auc = roc_auc_score(y_data, clf.predict_proba(x_data)[:,1])

	print("For the THRESHOLD {}, the negative predictive value is {:.4f}, the positive predictive value is {:.4f}, the auc is {:.4f}".format(threshold,neg_predictive_value, pos_predictive_value, roc_auc))

	return tn, fp, fn, tp





