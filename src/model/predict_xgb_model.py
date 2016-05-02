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
cxgb = joblib.load(utils.model_path + 'cxgb_all_' + parameter_str +'.pkl')

test = joblib.load(utils.processed_data_path + 'test_all_' + parameter_str +'.pkl')
X_test = test.ix[:,1:]

X_test.fillna(-1, inplace=True)

print "predict XGBClassifier..."

probs = cxgb.predict_proba(X_test)
sorted_index = np.argsort(-np.array(probs))[:,:5]

result = pd.DataFrame(columns = {'hotel_cluster'})

result['hotel_cluster'] = np.array([np.array_str(sorted_index[i])[1:-1] for i in range(sorted_index.shape[0])])

result.hotel_cluster.to_csv(utils.model_path + 
	'results/submission_cxgb_all_' + parameter_str + '.csv', header=True, index_label='id')


