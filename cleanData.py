import pandas as pd
import numpy as np
def cleanData(fileName):
	#loading the CSV File
	data = pd.read_csv(fileName)
	#renaming the attributes for interprability
	data = data.rename(columns={'CS_SITESPECIFIC_FACTOR_1': "DEPTH", 'CS_SITESPECIFIC_FACTOR_2': 'ULCERATION', 'CS_SITESPECIFIC_FACTOR_7': 'MITOSES', 'CS_SITESPECIFIC_FACTOR_3': 'CS_LYMPH_NODE_METS'})
	#attribs = ['AGE', 'SEX', 'RACE', 'SPANISH_HISPANIC_ORIGIN', 'DEPTH', 'ULCERATION','MITOSES', 'CS_EXTENSION', 'PRIMARY_SITE', 'CS_LYMPH_NODE_METS']  
	attribs = [ 'AGE','SEX', 'DEPTH', 'ULCERATION','MITOSES', 'CS_EXTENSION', 'PRIMARY_SITE', 'CS_LYMPH_NODE_METS']  

	#removing rows with NAN or missing data
	data = data[attribs].dropna()

	#More specific cleaning: 999 is often marked as the missing value; 980 also sometimes marked for missing data; 9 for hispanic origin; 988 for mitoses
	new_df = data[(data != 999).all(1)]
	new_df = new_df[new_df['DEPTH'] != 0]
	new_df = new_df[new_df['DEPTH'] <= 980]

	#new_df = new_df[new_df['SPANISH_HISPANIC_ORIGIN'] != 9]
	new_df = new_df[new_df['MITOSES'] <= 11]

	#Combining features with the same meaning -- coding schema have changed so there are values that mean the same thing
	new_df['ULCERATION'] = new_df['ULCERATION'].replace(10.0, 1.0)

	#indicates Clark Level
	new_df['CS_EXTENSION'] = new_df['CS_EXTENSION'].replace([310, 315, 320, 330, 335, 340, 350, 355, 360, 370, 375, 380, 400, 800, 950], [999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999])
	new_df = new_df[(new_df != 999).all(1)]

	#target
	new_df['CS_LYMPH_NODE_METS'] = new_df['CS_LYMPH_NODE_METS'].replace([5, 10],[0, 1])
	new_df = new_df[new_df['CS_LYMPH_NODE_METS'] <= 1.5]
	#new_df['CS_LYMPH_NODE_METS'].value_counts()


	#race (maybe get rid of?)
	#a = np.array(new_df['RACE'].values.tolist())
	#new_df['RACE'] = np.where(a > 2, 3, a).tolist()

	#a = np.array(new_df['SPANISH_HISPANIC_ORIGIN'].values.tolist())
	#new_df['SPANISH_HISPANIC_ORIGIN'] = np.where(a > 1, 1, a).tolist()
	
	#location of tumor - divided into head, trunk, limbs
	#new_df['PRIMARY_SITE'].value_counts()
	new_df['PRIMARY_SITE'] = new_df['PRIMARY_SITE'].replace(['C440', 'C441', 'C442', 'C443', 'C444', 'C445', 'C446', 'C447', 'C448', 'C449'], [0,0,0,0,0,1, 2,2,999,999])

	new_df = new_df[(new_df != 999).all(1)]


	#attribs = ['AGE', 'SEX', 'DEPTH', 'ULCERATION','RACE', 'SPANISH_HISPANIC_ORIGIN','MITOSES', 'CS_EXTENSION', 'PRIMARY_SITE']
	features = ['AGE','SEX', 'DEPTH', 'ULCERATION','MITOSES', 'CS_EXTENSION', 'PRIMARY_SITE']

	target = 'CS_LYMPH_NODE_METS'
	return new_df[features], new_df[target]
