from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
def classifierEvaluation(clf, threshold, X_data, y_data, classifier_name, data_descriptor):
	#creating predictions using classifier and a threshold for classifying as the positive class
	y_pred = np.where(clf.predict_proba(X_data)[:,1] > threshold, 1, 0)

	cf_mat = confusion_matrix(y_data, y_pred)
	tp, fp, fn, tn = cf_mat.ravel()
	#calculating the various measures of fit
	score = accuracy_score(y_data, y_pred)
	f1 = f1_score(y_data,y_pred)
	recall = recall_score(y_data, y_pred)
	precision = precision_score(y_data,y_pred)
	specificity = tn/(tn + fp)
	print(f"For the {data_descriptor} using {classifier_name}, the accuracy is {score}, the precision is {precision}, the recall is {recall}, the specificity is {specificity} and the f1 score is {f1}")

	group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
	group_counts = ["{0:0.0f}".format(value) for value in cf_mat.flatten()]
	group_perct = ["{0:.2%}".format(value) for value in cf_mat.flatten()/np.sum(cf_mat)]

	labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_perct)]
	labels = np.asarray(labels).reshape(2,2)

	sns.heatmap(cf_mat, annot=labels, fmt='', cmap = 'Blues')
	plt.show()