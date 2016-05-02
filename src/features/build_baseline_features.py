import numpy as np
import pandas as pd
import sys
import os
from sklearn.externals import joblib

scriptpath = os.path.dirname(os.path.realpath(sys.argv[0])) + '/../'
sys.path.append(os.path.abspath(scriptpath))
import utils


# time based features
def time_features_enricher(dataset):
	dataset['date_time_dt'] = pd.to_datetime(dataset.date_time, format = '%Y-%m-%d %H:%M:%S')
	dataset['date_time_dow'] = dataset.date_time_dt.dt.dayofweek
	dataset['date_time_hour'] = dataset.date_time_dt.dt.hour
	dataset['date_time_month'] = dataset.date_time_dt.dt.month

	dataset.loc[dataset.srch_ci == '2161-10-00', 'srch_ci'] = '2016-01-20'

	dataset['srch_ci_dt'] = pd.to_datetime(dataset.srch_ci, format = '%Y-%m-%d')
	dataset['srch_ci_dow'] = dataset.srch_ci_dt.dt.dayofweek
	dataset['srch_ci_month'] = dataset.srch_ci_dt.dt.month

	dataset['srch_co_dt'] = pd.to_datetime(dataset.srch_co, format = '%Y-%m-%d')
	dataset['srch_co_dow'] = dataset.srch_co_dt.dt.dayofweek
	dataset['srch_co_month'] = dataset.srch_co_dt.dt.month

	dataset['booking_window'] = (dataset['srch_ci_dt'] - dataset['date_time_dt'])/np.timedelta64(1, 'D')
	dataset['booking_window'].fillna(1000, inplace=True)
	dataset['booking_window'] = map(int, dataset['booking_window'])

	dataset['length_of_stay'] = (dataset['srch_co_dt'] - dataset['srch_ci_dt'])/np.timedelta64(1, 'D')


def gen_top_one_hot_encoding(row, field_name, top_vals):
	"""
    The helper function for gen_all_top_one_hot_encoding for one row
    :param row: the result instance to be filled
    :param field_name: another group by model result to merge to the row if the row has less than 5 hotel clusters
    :return: the updated hotel clusters for this instance
    """
	encoding = np.empty(len(top_vals))
	encoding.fill(0)

	encoding[top_vals==row[field_name]] = 1
	return pd.Series(encoding)


def gen_top_one_hot_encoding_column(dataset, field_name, top_vals):
	"""
    Generate top 10 categorical one hot encoding for one field, based on the analysis shown in the reports/figures/report.html
    :param dataset: the result instance to be filled
    :param field_name: another group by model result to merge to the row if the row has less than 5 hotel clusters
    :return: the updated hotel clusters for this instance
    """
	encoding = dataset.apply(lambda row: gen_top_one_hot_encoding(row, field_name, np.array(top_vals)), axis=1)
	top_vals_str = map(str, top_vals)

	encoding.columns = map('_'.join, zip([field_name] * len(top_vals), top_vals_str))

	return encoding

def gen_all_top_one_hot_encoding_columns(dataset):
	site_name_top_vals = [2, 11, 37, 24, 34, 8, 13, 23, 17, 28]
	site_name_encoding = gen_top_one_hot_encoding_column(dataset, 'site_name', site_name_top_vals)

	posa_continent_top_vals = [3, 1, 2, 4, 0]
	posa_continent_encoding = gen_top_one_hot_encoding_column(dataset, 'posa_continent', posa_continent_top_vals)

	user_location_country_top_vals = [66, 205, 69, 3, 77, 46, 1, 215, 133, 68]
	user_location_country_encoding = gen_top_one_hot_encoding_column(dataset, 'user_location_country', user_location_country_top_vals)

	user_location_region_top_vals = [174, 354, 348, 442, 220, 462, 155, 135, 50, 258]
	user_location_region_encoding = gen_top_one_hot_encoding_column(dataset, 'user_location_region', user_location_region_top_vals)

	channel_top_vals = [9, 10, 0, 1, 5, 2, 3, 4, 7, 8]
	channel_encoding = gen_top_one_hot_encoding_column(dataset, 'channel', channel_top_vals)

	srch_destination_type_id_top_vals = [1, 6, 3, 5, 4, 8, 7, 9, 0]
	srch_destination_type_id_encoding = gen_top_one_hot_encoding_column(dataset, 'srch_destination_type_id', srch_destination_type_id_top_vals)

	hotel_continent_top_vals = [2, 6, 3, 4, 0, 5, 1]
	hotel_continent_encoding = gen_top_one_hot_encoding_column(dataset, 'hotel_continent', hotel_continent_top_vals)

	hotel_country_top_vals = [50, 198, 70, 105, 8, 204, 77, 144, 106, 63]
	hotel_country_encoding = gen_top_one_hot_encoding_column(dataset, 'hotel_country', hotel_country_top_vals)

	return site_name_encoding, posa_continent_encoding, user_location_country_encoding, user_location_region_encoding, \
	channel_encoding, srch_destination_type_id_encoding, hotel_continent_encoding, hotel_country_encoding

def fill_na_features(dataset):
	dataset.fillna(-1, inplace=True)


#############################################################
####################   train dataset     ####################
#############################################################

train = utils.load_train('baseline')

# train = pd.read_csv(utils.raw_data_path + 'train_1000.csv',
#                         dtype={'date_time':str, 'is_booking':np.int8, 'site_name':np.int32,'posa_continent':np.int32, 'user_location_country':np.int32, \
#                         'user_location_region':np.int32, 'orig_destination_distance':np.double, \
#                         'is_mobile':np.int8, 'is_package':np.int8, 'channel':np.int32, 'srch_ci':str, 'srch_co':str, \
#                         'srch_adults_cnt':np.int32, 'srch_children_cnt':np.int32, 'srch_rm_cnt':np.int32, \
#                         'srch_destination_type_id':np.int32, 'hotel_continent':np.int32, 'hotel_country':np.int32, \
#                         'hotel_cluster':np.int32},
#                         usecols=['date_time', 'is_booking', 'site_name', 'posa_continent', 'user_location_country', \
#                         'user_location_region', 'orig_destination_distance', \
#                         'is_mobile', 'is_package', 'channel', 'srch_ci', 'srch_co', \
#                         'srch_adults_cnt', 'srch_children_cnt', 'srch_rm_cnt', \
#                         'srch_destination_type_id', 'hotel_continent', 'hotel_country', \
#                         'hotel_cluster'])

train_is_booking = train[train.is_booking == 1]
train_is_booking.reset_index(inplace = True)
train_is_booking.is_copy = False
del train

print 'generate train time features...'
time_features_enricher(train_is_booking)

print 'generate train one hot encoding features...'
site_name_encoding, posa_continent_encoding, user_location_country_encoding, user_location_region_encoding, \
	channel_encoding, srch_destination_type_id_encoding, hotel_continent_encoding, hotel_country_encoding = \
	gen_all_top_one_hot_encoding_columns(train_is_booking)

print 'fill train na features...'
fill_na_features(train_is_booking)

print 'concat all train baseline features...'
train_is_booking_features = pd.concat([train_is_booking[['hotel_cluster', 'date_time', 'orig_destination_distance', \
	'is_mobile', 'is_package', 'srch_adults_cnt', 'srch_children_cnt', 'srch_rm_cnt', \
	'date_time_dow', 'date_time_hour', 'date_time_month', 'srch_ci_dow', 'srch_ci_month', \
	'srch_co_dow', 'srch_co_month', 'booking_window', 'length_of_stay']], \
	site_name_encoding, posa_continent_encoding, user_location_country_encoding, user_location_region_encoding, \
	channel_encoding, srch_destination_type_id_encoding, hotel_continent_encoding, hotel_country_encoding], axis=1)

train_is_booking_features.to_csv(utils.processed_data_path +
	'_'.join(['train_is_booking_baseline', 'year', utils.train_year]) + '.csv', 
	header=True, index=False)

del train_is_booking

#############################################################
####################   test dataset      ####################
#############################################################

test  = utils.load_test('baseline')

# test = pd.read_csv(utils.raw_data_path + 'test_1000.csv',
#                 dtype={'date_time':str, 'site_name':np.int32,'posa_continent':np.int32, 'user_location_country':np.int32, \
#                 'user_location_region':np.int32, 'orig_destination_distance':np.double, \
#                 'is_mobile':np.int8, 'is_package':np.int8, 'channel':np.int32, 'srch_ci':str, 'srch_co':str, \
#                 'srch_adults_cnt':np.int32, 'srch_children_cnt':np.int32, 'srch_rm_cnt':np.int32, \
#                 'srch_destination_type_id':np.int32, 'hotel_continent':np.int32, 'hotel_country':np.int32},
#                 usecols=['date_time', 'site_name', 'posa_continent', 'user_location_country', \
#                 'user_location_region', 'orig_destination_distance', \
#                 'is_mobile', 'is_package', 'channel', 'srch_ci', 'srch_co', \
#                 'srch_adults_cnt', 'srch_children_cnt', 'srch_rm_cnt', \
#                 'srch_destination_type_id', 'hotel_continent', 'hotel_country'])
print 'generate test time features...'
time_features_enricher(test)

print 'generate test one hot encoding features...'
site_name_encoding, posa_continent_encoding, user_location_country_encoding, user_location_region_encoding, \
	channel_encoding, srch_destination_type_id_encoding, hotel_continent_encoding, hotel_country_encoding = \
	gen_all_top_one_hot_encoding_columns(test)

print 'fill test na features...'
fill_na_features(test)

print 'concat all test baseline features...'
test_features = pd.concat([test[['date_time', 'orig_destination_distance', \
	'is_mobile', 'is_package', 'srch_adults_cnt', 'srch_children_cnt', 'srch_rm_cnt', \
	'date_time_dow', 'date_time_hour', 'date_time_month', 'srch_ci_dow', 'srch_ci_month', \
	'srch_co_dow', 'srch_co_month', 'booking_window', 'length_of_stay']], \
	site_name_encoding, posa_continent_encoding, user_location_country_encoding, user_location_region_encoding, \
	channel_encoding, srch_destination_type_id_encoding, hotel_continent_encoding, hotel_country_encoding], axis=1)

test_features.to_csv(utils.processed_data_path +
	'_'.join(['test_baseline', 'year', utils.train_year]) + '.csv', 
	header=True, index=False)
