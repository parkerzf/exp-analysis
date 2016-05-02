import numpy as np
import pandas as pd
import sys
import os
from sklearn.externals import joblib
import xgboost as xgb

scriptpath = os.path.dirname(os.path.realpath(sys.argv[0])) + '/../'
sys.path.append(os.path.abspath(scriptpath))
import utils

train = joblib.load(utils.processed_data_path + 'train_is_booking_all_top_5_cw_0.05_year_all.pkl')

X_train = train.ix[:,2:]
y_train = pd.DataFrame(train['hotel_cluster'].astype(int))


print "train XGBClassifier..."
cxgb = xgb.XGBClassifier(max_depth=15, n_estimators=100, learning_rate=0.02, colsample_bytree=0.5, min_child_weight=5, verbose=1)

cxgb.fit(X_train, y_train.ravel())
joblib.dump(cxgb, utils.model_path + 'cxgb_all_without_time_top_5_cw_0.05_year_all.pkl')





