import numpy as np
import pandas as pd
import sys
import os
from sklearn.externals import joblib
import xgboost as xgb

scriptpath = os.path.dirname(os.path.realpath(sys.argv[0])) + '/../'
sys.path.append(os.path.abspath(scriptpath))
import utils

parameter_str = '_'.join(['top', str(utils.k), 'cw', str(utils.click_weight), 'year', utils.train_year])
train = joblib.load(utils.processed_data_path + 'train_is_booking_all'+ parameter_str +'.pkl')

X_train = train.ix[:,2:]
y_train = train['hotel_cluster'].astype(int)


print "train XGBClassifier..."
cxgb = xgb.XGBClassifier(max_depth=15, n_estimators=50, learning_rate=0.1, colsample_bytree=0.5, min_child_weight=5)

cxgb.fit(X_train, y_train, verbose=True)
joblib.dump(cxgb, utils.model_path + 'cxgb_all'+ parameter_str + '.pkl')





