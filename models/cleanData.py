import pandas as pd
import numpy as np
def cleanData(data):
	#what attributes we want to use (that don't involve data leakage)
	attribs = [ 'AGE','SEX', 'DEPTH', 'ULCERATION','MITOSES', 'CS_EXTENSION', 'PRIMARY_SITE', 'CS_LYMPH_NODE_METS']  

	#removing rows with NAN or missing data
	data = data[attribs].dropna()

	#More specific cleaning: 999 is often marked as the missing value; don't want to include melanoma of depth 0; 988 for mitoses
	new_df = data[(data != 999).all(1)]
	new_df = new_df[(new_df != 1022).all(1)]
	new_df['DEPTH'] = new_df['DEPTH'].replace(989.0, 980.0)
	new_df = new_df[new_df['DEPTH'] <= 987]
	new_df = new_df[new_df['DEPTH'] != 0]
	new_df = new_df[new_df['MITOSES'] != 988]
	

	#replacing old codings with the current standards
	new_df['MITOSES'] = new_df['MITOSES'].replace([990, 991], [0, 1])
	new_df = new_df[new_df['MITOSES'] <= 11]

	#Combining features with the same meaning -- coding schema have changed so there are values that mean the same thing
	new_df['ULCERATION'] = new_df['ULCERATION'].replace(10.0, 1.0)
	new_df = new_df[new_df['ULCERATION']<=1]

	#indicates Clark Level, the numbers we are replacing with 999 are coded for stage of cancer, which soley depends on depth
	#and whether there is ulceration and not how far into the dermis it has invaded, so we exclude those values from consideration
	new_df['CS_EXTENSION'] = new_df['CS_EXTENSION'].replace([
    	310, 315, 320, 330, 335, 340, 350, 355, 360, 370, 375, 380, 400, 950],
    	[999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999])
	new_df = new_df[(new_df != 999).all(1)]

	#target + recoding old coldings to match the current standard
	new_df['CS_LYMPH_NODE_METS'] = new_df['CS_LYMPH_NODE_METS'].replace([5, 10],[0, 1])
	new_df = new_df[new_df['CS_LYMPH_NODE_METS'] <= 1.5]

	#binning the various locations into Head (0), Trunk (1), and Limbs (2)
	#we recode both C44x and 44x as the coding differed between the NCDB and SEER databases
	new_df['PRIMARY_SITE'] = new_df['PRIMARY_SITE'].replace(['C440', 'C441', 'C442', 'C443', 'C444', 'C445', 'C446', 'C447', 'C448', 'C449',
		'440', '441', '442', '443','444','445','446','447','448','449'], [0,0,0,0,0,1, 2,2,999,999, 0,0,0,0,0,1, 2,2,999,999])

	new_df = new_df[(new_df != 999).all(1)]

	features = ['AGE','SEX', 'DEPTH', 'ULCERATION','MITOSES', 'CS_EXTENSION', 'PRIMARY_SITE']
	return new_df[features], new_df['CS_LYMPH_NODE_METS']
