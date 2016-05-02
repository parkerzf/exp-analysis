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
#X_test = test.ix[:,1:]
X_test = test.ix[:999,1:]

print "predict RandomForest Classifier..."

probs = cforest.predict_proba(X_test)

print probs.shape

best_5 = np.argsort(probs, axis=0)[-5:]

print best_5.shape


print cforest.predict(X_test)



#TODO compute the top 5 hotel clusters


