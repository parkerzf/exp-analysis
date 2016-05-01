import numpy as np
import pandas as pd
import sys
import os
from sklearn.externals import joblib

scriptpath = os.path.dirname(os.path.realpath(sys.argv[0])) + '/../'
sys.path.append(os.path.abspath(scriptpath))
import utils

try:
    test = joblib.load(utils.processed_data_path + 'test_group_by.pkl')
except:
	print 'load raw test data'
	test = pd.read_csv(utils.raw_data_path + 'test.csv',
                 	dtype={'srch_destination_id':np.int32, 'hotel_market':np.int32, \
                 	'orig_destination_distance':np.double, 'user_id':np.int32},
                    usecols=['srch_destination_id', 'hotel_market', \
                    'orig_destination_distance', 'user_id'])
	joblib.dump(test, utils.processed_data_path + 'test_group_by.pkl')

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

def fill_na(prev_result, group_by_field):
	"""
	Use another group by model to enrich the previous model result for empty result instances
	:param prev_result: the dataframe of the previous model result
	:param group_by_field: group by field to get the another group by model
	:return: the dataframe with the updated model result
	"""
	group_by_model = joblib.load(utils.model_path + \
		'_'.join(['top', str(utils.k), 'cw', str(utils.click_weight), 'group', group_by_field, 'year', utils.train_year]) + '.pkl')
	merged_df = test.merge(group_by_model, how='left',left_on= group_by_field, right_index=True)
	new_result = prev_result.hotel_cluster.fillna(merged_df.hotel_cluster, inplace=True)

def fill_top_5(row, merged_df):
	"""
	The helper function for fill_all_top_5 function to merging a single result instance with another group by model result
	:param row: the result instance to be filled
	:param merged_df: another group by model result to merge to the row if the row has less than 5 hotel clusters
	:return: the updated hotel clusters for this instance
	"""
	if not isinstance(row[1], float):
			length = len(row[1].split())
			if length < 5:
				candidates = merged_df.hotel_cluster[row[0]]
				if not isinstance(candidates, float):
					return row[1] + ' ' + ' '.join (candidates.split()[:5 - length])
				else:
					return row[1]
			else:
				return row[1]
	else:
		return merged_df.hotel_cluster[row[0]]


def fill_all_top_5(prev_result, group_by_field):
	"""
	Use another group by model to enrich the previous model result for result instances with less than 5 hotel clusters
	:param prev_result: the dataframe of the previous model result
	:param group_by_field: group by field to get the another group by model
	:return: the dataframe with the updated model result
	"""
	group_by_model = joblib.load(utils.model_path + \
		'_'.join(['top', str(utils.k), 'cw', str(utils.click_weight), 'group', group_by_field, 'year', utils.train_year]) + '.pkl')
	merged_df = test.merge(group_by_model, how='left',left_on= group_by_field, right_index=True)

	prev_result['new_hotel_cluster'] = prev_result.apply(lambda row: fill_top_5(row, merged_df), axis=1)
	new_result = prev_result.drop('hotel_cluster', 1)
	new_result.columns = ['id', 'hotel_cluster']

	return new_result


print 'predict with orig_destination_distance model...'
result = predict_group_by_model('orig_destination_distance')
print 'predict with srch_destination_id model...'
result = fill_all_top_5(result, 'srch_destination_id')
print 'predict with user_id model...'
result = fill_all_top_5(result, 'user_id')
print 'predict with hotel_market model...'
result = fill_all_top_5(result, 'hotel_market')

result.hotel_cluster.to_csv(utils.model_path + 
	'results/submission_' + '_'.join(['top', str(utils.k), 'cw', str(utils.click_weight), 'year', utils.train_year]) + '.csv', 
	header=True, index_label='id')

