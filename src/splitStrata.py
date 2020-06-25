import pandas as pd

def splitStrata(X_data, y_data):
	"""This function takes X data and y data and divides it into the various cancer stages.
	"""
	tmp = X_data
	tmp['TARGET'] = y_data
	c7 = tmp['DEPTH'] < 80
	c8 = tmp['DEPTH'] >= 80
	c1 = tmp['DEPTH'] <= 100
	c2 = tmp['ULCERATION'] == 0
	c4 = (tmp['DEPTH'] > 100) & (tmp['DEPTH'] <= 200)
	c5 = (tmp['DEPTH'] > 200) & (tmp['DEPTH'] <= 400)
	c6 = tmp['DEPTH'] > 400

	T1a = tmp[c7 & c2]
	y1a = T1a['TARGET']
	T1a = T1a.drop(columns=["TARGET"])
	
	T1b = tmp[(c1 & ~(c2) )| (c8 & c1)]
	y1b = T1b['TARGET']
	T1b = T1b.drop(columns=["TARGET"])

	
	T2a = tmp[c4 & c2]
	y2a = T2a['TARGET']
	T2a = T2a.drop(columns=["TARGET"])


	T2b = tmp[c4 & ~(c2)]
	y2b = T2b['TARGET']
	T2b = T2b.drop(columns=["TARGET"])


	T3a = tmp[c5 & c2]
	y3a = T3a['TARGET']
	T3a = T3a.drop(columns=["TARGET"])

	T3b = tmp[c5 & ~(c2)]
	y3b = T3b['TARGET']
	T3b = T3b.drop(columns=["TARGET"])


	T4a = tmp[c6 & ~(c2)]
	y4a = T4a['TARGET']
	T4a = T4a.drop(columns=["TARGET"])

	T4b = tmp[c6 & (c2)]
	y4b = T4b['TARGET']
	T4b = T4b.drop(columns=["TARGET"])

	return T1a, y1a, T1b, y1b, T2a, y2a, T2b, y2b, T3a, y3a, T3b, y3b, T4a, y4a, T4b, y4b