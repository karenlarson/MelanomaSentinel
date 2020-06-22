# MelanomaSentinel

The goal of this project is to create a web-application that doctors can use to advise patients on whether they need a sentinel lymph node biopsy (SLNB). Currently, only about 10% of SLNB are necessary, leading to additional health complications and financial costs for those that undergo an unneeded treatment. This repo provides the code used to train and test the logistic regression model used, as well as deploy the web-application.

# Motivation
Melanoma has a 97% 5-year survival rate when it stays localized, but that decreases to less than 60% for patients that have spread to the local lymph nodes.

The sentinel lymph node is the most likely first node for melanoma to have spread, but currently the recommendation for a SLNB is if there is at least a 5% chance of the cancer having spread to the sentinel lymph nodes. This leads to only 10-20% of all SLNB leading to a positive test. SLNB have around a 10% chance for complications, so reducing the number of unneeded SLNB would help improve health outcomes for patients and also reduce unneeded costs.

To this end, this application provides an improved predictor for SLNB and reduces the number of unnecessary tests by around 16% while maintaining a fairly high recall (99.6%).

# Project Information
This project was done in conjuction with the [Cutaneous Oncology Research Lab at University Hospitals](https://www.uhhospitals.org/doctors/Yu-Wesley-1609244086). My contribution was creating SQL databases, gathering the NIH SEER data, exploring the data, creating and comparing prediction models, and reporting the results. This was done during the [Summer 2020 Insight Boston](https://insightfellows.com/health-data) session.

# Step 0: The Driver of the Code

The main function to create and train the logistic regression model is stored in MelanomaSentinel.py. This requires a SQL database of the melanoma patient data, cleanData.py, encodeAttribs.py, baseLine.py, and classifierEvaluation.py contained in this repo. This code assumes that you have pandas, scikit-learn, pickle, and postgresql installed. The required enviornment is included in the associated environment.yml file and in the requirements.txt.

# Step 1: Merge Data Sources
The data from this project were taken from the [National Cancer Database](https://www.facs.org/quality-programs/cancer/ncdb) and the [NIH's SEER Program](https://seer.cancer.gov/data/). 

The initial data were in .csv files, which I transferred to a SQL database for faster access.

Python script used to clean the data is located in cleanData.py.

The data for this project are not publically available, so I am not able to provide links to the csv files or to the SQL database I created.

# Step 2: Feature Engineering

For the final product, I used the following numerical attributes:

- Breslow Thickness (Melanoma Depth)
- Age
- Clark Level (What level of the skin has the melanoma invaded)
- Mitotic Rate
- Ulceration (1 if present, 0 otherwise)
- Sex (2 for female, 1 for male)
- Primary Site of the Tumor (0 for Head, 1 for Trunk, 2 for Limbs)
- Presence of Tumor Metastasis (Target Variable)

The first four I considered numerical attributes, while the last three I considered numerical attributes. For the numerical attributes, the key indicator of melanoma extent medically is its depth. However, melanoma are three-dimensional objects, so I believed that using information about it's surface area and volume could provide valueable predictive power for the model. Therefore, I considered looking at polynomial features, specifically polynomial features of degree 3 in order to account for these considerations. 

The categorical features were one-hot encoded, leading to a total of 42 features trained for the model.

The pipeline used is incldued in encodeAttributes.py. To run the pipeline, import preprocessing\_pipeline(numerical\_attribs, polynomial\_degree, categorical\_attributes) from encodeAttributes.py, where numerical\_attributes is a list of the columns with numerical values, polynomial\_degree is the degree of polynomial needed for the model, and categorical\_attributes is the list of the columns with categorical values.

# Step 3: Train Models
In this final version of the code, I use a baseline estimator and a logistic regression classifier. The baseline estimator just predicts accuracy, which is not the best metric of success for a skewed dataset like this, as the specificity will be 1.00, but it would miss every single biopsy case.

The final version to fit the model is in the MelanomaSentinel.py file, which trains and pickles the trained model to logistic_clf.sav. To evaluate the goodness of the classifier for various thresholds, we call a function from classifierEvaluation.py, which takes a classifier, the probability cut-off thresholds, features, targets, and two strings indicating the type of classifier and the type of data (train, validation, test). The strings are used to print output to screen and create titles for confusion matricies, to make it easier to evaluate and save the results for various thresholds.

# Step 4: Streamlit App

The final tool is built as a streamlit application in sentinel_app.py, available at [melanoma-sentinel.herokuapp.com](https://melanoma-sentinel.herokuapp.com). To run the app locally on a machine type this in the terminal:

`streamlit run sentinel_app.py`

The requirements for the app are included in requirements.txt and the conda environment used is in environment.yml.

To run the app, sentinel\_app.py, encodeAttribs.py, logistic\_clf.sav and transformer.sav are required.

# Presentation

Google slides summarizing the project can be found [here](https://docs.google.com/presentation/d/1--eW4tCH3lwxLpfyjghiqK3en7VOY016BZvjH87k4mw/edit?usp=sharing).