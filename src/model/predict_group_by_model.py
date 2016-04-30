import numpy as np
import pandas as pd
from sklearn.externals import joblib

import sys
import os
scriptpath = os.path.dirname(os.path.realpath(sys.argv[0])) + '/../'
sys.path.append(os.path.abspath(scriptpath))
import utils
from utils import *

test = pd.read_csv(utils.raw_data_path + 'test.csv',
                 	dtype={'srch_destination_id':np.int32, 'hotel_market':np.int32, \
                 	'orig_destination_distance':np.double, 'user_id':np.int32},
                    usecols=['srch_destination_id', 'hotel_market', \
                    'orig_destination_distance', 'user_id'])

def predict_group_by_model(group_by_field):
	group_by_model = joblib.load(utils.model_path + 
		'_'.join(['top', str(utils.k), 'cw', str(utils.click_weight), 'group', group_by_field, 'year', utils.train_year]) + '.pkl')
	merged_df = test.merge(group_by_model, how='left',left_on= group_by_field, right_index=True)

	merged_df.reset_index(inplace = True)
	return merged_df[['index', 'hotel_cluster']]

def fill_na(pre_result, group_by_field):
	group_by_model = joblib.load(utils.model_path + \
		'_'.join(['top', str(utils.k), 'cw', str(utils.click_weight), 'group', group_by_field, 'year', utils.train_year]) + '.pkl')
	merged_df = test.merge(group_by_model, how='left',left_on= group_by_field, right_index=True)
	pre_result.hotel_cluster.fillna(merged_df.hotel_cluster, inplace=True)

def fill_top_5(row, merged_df):
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


def fill_all_top_5(pre_result, group_by_field):
	group_by_model = joblib.load(utils.model_path + \
		'_'.join(['top', str(utils.k), 'cw', str(utils.click_weight), 'group', group_by_field, 'year', utils.train_year]) + '.pkl')
	merged_df = test.merge(group_by_model, how='left',left_on= group_by_field, right_index=True)

	pre_result['new_hotel_cluster'] = pre_result.apply(lambda row: fill_top_5(row, merged_df), axis=1)
	pre_result = pre_result.drop('hotel_cluster', 1)
	pre_result.columns = ['id', 'hotel_cluster']

	return pre_result


print 'predict with orig_destination_distance model...'
merged_df = predict_group_by_model('orig_destination_distance')
print 'predict with srch_destination_id model...'
merged_df = fill_all_top_5(merged_df, 'srch_destination_id')
print 'predict with user_id model...'
merged_df = fill_all_top_5(merged_df, 'user_id')
print 'predict with hotel_market model...'
merged_df = fill_all_top_5(merged_df, 'hotel_market')

merged_df.hotel_cluster.to_csv(utils.model_path + 
	'results/submission_' + '_'.join(['top', str(utils.k), 'cw', str(utils.click_weight), 'year', utils.train_year]) + '.csv', 
	header=True, index_label='id')

