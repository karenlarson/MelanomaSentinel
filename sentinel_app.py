import streamlit as st
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from encodeAttribs import preprocess_pipeline
import numpy as np
import matplotlib.pyplot as plt
from pywaffle import Waffle


log_clf = pickle.load(open('logistic_clf.sav','rb'))
#gnb_clf = pickle.load(open('gnb_clf.sav','rb'))
pipe = pickle.load(open('transform.sav', 'rb'))

st.title('Welcome to MelanomaSentinel')
st.subheader('*A doctor\'s second opinion*')
st.write('Enter the following demographics for the patient in the left panel for their probability of needing a sentinal lymph node biopsy:')
st.write('Age, Breslow Thickness (in mm), Clark Level, Mitotic Count, Presence of Ulceration, Sex, and Primary Site of Melanoma')

#sidebars to get user input
res = { 'DEPTH': 0, 'CS_EXTENSION': 0, 'MITOSES': 0, 'SEX': 0, 'ULCERATION': 0, 'PRIMARY_SITE': 0}
res['AGE'] = st.sidebar.number_input('Age of patient:', min_value= 0, max_value=120, value=40, step=10)
res['DEPTH'] = 100*st.sidebar.number_input('Measured thickness (depth) of melanoma (in millimeters):', min_value = 0.0, max_value=9.80, value=1.00,step = 0.1)
res['CS_EXTENSION'] = 100*(st.sidebar.number_input('Clark level (1 - 5): ', min_value = 1, max_value = 5, value = 1, step = 1) - 1)
res['MITOSES'] = st.sidebar.number_input('Primary tumor mitotic count (# mitoses per square milimeter (mm)): ', min_value = 0, max_value = 11, value = 1, step = 1)
res['ULCERATION'] = 0 + (st.sidebar.text_input('Presence of ulceration (Yes or No):', 'No')[0].lower() == 'y')
holder = st.sidebar.text_input('Sex:', 'Male')
res['SEX'] = 1 + (holder[0].lower() == 'f')
holder = st.sidebar.text_input('Primary site of melanoma (Head, Trunk, Limbs):', 'Head')
if holder[0].lower() == 'h':
	res['PRIMARY_SITE'] = 0
elif holder[0].lower() == 't':
	res['PRIMARY_SITE'] = 1
else:
	res['PRIMARY_SITE'] = 2

#storing data in a dataframe
res_df = pd.DataFrame(res, index = [0])

#transforming new input and making predictions (both class and the probabilities)
x_trans = pipe.transform(res_df)
y_pred_log = log_clf.predict(x_trans)
y_proba_log = log_clf.predict_proba(x_trans)
st.subheader("The chance that a person with these demographics would have a positive sentinel lymph node biopsy is {:.2f}%.".format(y_proba_log[0,1]*100))

#visualization to aid in understanding
#labels = 'Negative Biopsy', 'Positive Biopsy'
sizes = y_proba_log
#explode = (0, 0.1)
#fig, ax1 = plt.subplots()
#ax1.pie(sizes, explode=explode, labels=labels, autopct="%1.1f%%", shadow=False, startangle=45)
#ax1.axis('equal')
#plt.show()
#st.pyplot()

if res['DEPTH'] <= 100:
	if res['MITOSES'] == 0 and res['ULCERATION'] == 0:
		stage = "T1a"
	else:
		stage = "T1b"
elif res['DEPTH'] <= 200:
	if res['ULCERATION'] == 0:
		stage = "T2a"
	else:
		stage = "T2b"
elif res['DEPTH'] <= 400:
	if res['ULCERATION'] == 0:
		stage = "T3a"
	else:
		stage = "T3b"

else:
	if res['ULCERATION'] == 0:
		stage = "T4a"
	else:
		stage = "T4b"

dat = {"Positive Biopsy": round(sizes[0, 1]*100), "Negative Biopsy": round(sizes[0, 0]*100)}

fig = plt.figure(FigureClass = Waffle, rows = 10, values = dat, icons = "child", icon_size=18, icon_legend = True, legend={'loc': 'upper left', 'bbox_to_anchor': (1, 1)})
#plt.show()
st.pyplot()

st.write("This number shows the probability that the melanoma has metastasized to other sites.")

st.write(f"Based upon the Breslow Thickness, ulceration, and mitosis, the patient's tumor is classification {stage}. Patients with this classifications have had the following outcomes")

#y_pred_gnb = gnb_clf.predict(x_trans)
#y_proba_gnb = gnb_clf.predict_proba(x_trans)
#st.write("Probability of having a negative biopsy (0) or positive biopsy (1):", y_proba_gnb)

