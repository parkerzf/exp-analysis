import numpy as np
import pandas as pd
import sys
import os
from sklearn.externals import joblib

scriptpath = os.path.dirname(os.path.realpath(sys.argv[0])) + '/../'
sys.path.append(os.path.abspath(scriptpath))
import utils

test  = utils.load_test('group_by')

def predict_group_by_model(group_by_field):
	"""
	Use group by model to predict the top 5 hotel clusters for the test data
	:param group_by_field: group by field to get the related group by model 
	:return: the dataframe with the submission format according to the model
	"""
	group_by_model = joblib.load(utils.model_path + 
		'_'.join(['top', str(utils.k), 'cw', str(utils.click_weight), 'group', group_by_field, 'year', utils.train_year]) + '.pkl')
	merged_df = test.merge(group_by_model, how='left',left_on= group_by_field, right_index=True)

	merged_df.reset_index(inplace = True)
	result = merged_df[['index', 'hotel_cluster']]
	result.columns =  ['id', 'hotel_cluster']
	return result


print 'predict with orig_destination_distance model...'
result = predict_group_by_model('orig_destination_distance')
print 'predict with srch_destination_id model...'
result = utils.fill_all_top_5(test, result, 'srch_destination_id')
print 'predict with user_id model...'
result = utils.fill_all_top_5(test, result, 'user_id')
print 'predict with hotel_market model...'
result = utils.fill_all_top_5(test, result, 'hotel_market')

result.hotel_cluster.to_csv(utils.model_path + 
	'results/submission_' + '_'.join(['top', str(utils.k), 'cw', str(utils.click_weight), 'year', utils.train_year]) + '.csv', 
	header=True, index_label='id')

