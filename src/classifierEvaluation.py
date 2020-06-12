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
	tn, fp, fn, tp = cf_mat.ravel()
	#calculating the various measures of fit
	score = accuracy_score(y_data, y_pred)
	f1 = f1_score(y_data,y_pred)
	recall = recall_score(y_data, y_pred)
	precision = precision_score(y_data,y_pred)
	specificity = tn/(tn + fp)
	print("For the {} using {}, the accuracy is {:.4f}, the precision is {:.4f}, the recall is {:.4f}, the specificity is {:.4f} and the f1 score is {:.4f}".format(data_descriptor, classifier_name, score, precision, recall, specificity, f1))

	group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
	group_counts = ["{0:0.0f}".format(value) for value in cf_mat.flatten()]
	group_perct = [0]*4
	group_perct[0] = "{0:.2%}".format(tn/(tn + fp))
	group_perct[1] = "{0:.2%}".format(fp/(tn + fp))
	group_perct[2] = "{0:.2%}".format(fn/(fn + tp))
	group_perct[3] = "{0:.2%}".format(tp/(fn + tp))
	#group_perct = ["{0:.2%}".format(value) for value in cf_mat.flatten()/np.sum(cf_mat)]

	labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_perct)]
	labels = np.asarray(labels).reshape(2,2)

	a = np.array([[tn/(tn + fp), fp/(tn + fp)],[fn/(fn + tp), tp/(fn + tp)]])
	print(cf_mat)

	sns.heatmap(a, annot=labels, fmt='', cmap = 'Blues')
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	title = "Confusion Matrix for {} using {}".format(classifier_name, data_descriptor)
	plt.title(title)
	plt.show()