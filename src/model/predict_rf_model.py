import numpy as np
import pandas as pd
import sys
import os
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier

scriptpath = os.path.dirname(os.path.realpath(sys.argv[0])) + '/../'
sys.path.append(os.path.abspath(scriptpath))
import utils

cforest = joblib.load(utils.model_path + 'rf_all_without_time_top_5_cw_0.05_year_all.pkl')

test = joblib.load(utils.processed_data_path + 'test_all_top_5_cw_0.05_year_all.pkl')
X_test = test.ix[:,1:]

print "train RandomForest Classifier..."
cforest = RandomForestClassifier(n_estimators=100, max_depth=50, min_samples_split=50, min_samples_leaf=5, 
	random_state=0, verbose = 1, n_jobs=-1)

pred_probas = cforest.predict_proba(X_test)


#TODO compute the top 5 hotel clusters


