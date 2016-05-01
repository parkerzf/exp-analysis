import numpy as np
import pandas as pd
import sys
import os
from sklearn.externals import joblib

scriptpath = os.path.dirname(os.path.realpath(sys.argv[0])) + '/../'
sys.path.append(os.path.abspath(scriptpath))
import utils


try:
    train = joblib.load(utils.processed_data_path + 'train_group_by.pkl')
except:
	print 'load raw train data'
	train = pd.read_csv(utils.raw_data_path + 'train.csv',
                 	dtype={'date_time':str, 'is_booking':bool,'srch_destination_id':np.int32, 'hotel_market':np.int32, \
                 	'orig_destination_distance':np.double, 'user_id':np.int32, 'hotel_cluster':np.int32},
                    usecols=['date_time', 'is_booking', 'srch_destination_id', 'hotel_market', \
                    'orig_destination_distance', 'user_id', 'hotel_cluster'])
	joblib.dump(train, utils.processed_data_path + 'train_group_by.pkl')


train_2013 = train[train.date_time < '2014-01-01 00:00:00']
train_2014 = train[train.date_time >= '2014-01-01 00:00:00']



def top_k_relevence(group, k = utils.k):
    """
	Order and get the topk hotel cluters by the relevance score in desc order
	:param group: the aggregate group with hotel cluster relevance scores
	:param k: the top k value
	:return: the topk hotel clusters for the aggregate group
	"""
    idx = group.relevance.nlargest(k).index
    top_k_relevence = group.hotel_cluster[idx].values
    return np.array_str(top_k_relevence)[1:-1]

def gen_top_k_group_by_model(group_by_field, click_weight = utils.click_weight, year = 'all'):
	"""
	Generate and dump the group by model with top k hotel clusters 
	:param group_by_field: group by field to generate the group with hotel cluster relevance scores
	:param click_weight: the weight for the clicks
	:param year: Year filter on training data
	:return: the topk group by model with respect to group_by_field
	"""
	dump_path = utils.model_path + \
		'_'.join(['top', str(utils.k), 'cw', str(utils.click_weight), 'group', group_by_field, 'year', year]) + '.pkl'


	if os.path.exists(dump_path):
		print 'file: ' + dump_path + ' exists!'
		return

	source = train
	if year == '2013':
		source = train_2013
	elif year == '2014':
		source = train_2014
	
	agg = source.groupby([group_by_field, 'hotel_cluster'])['is_booking'].agg(['sum','count'])
	agg['count'] -= agg['sum']
	agg = agg.rename(columns = {'sum':'bookings','count':'clicks'})
	agg['relevance'] = agg['bookings'] + click_weight * agg['clicks'] # the weighted sum of bookings count and clicks count
	agg.reset_index(inplace = True)
	top_clusters = agg.groupby([group_by_field]).apply(top_k_relevence)
	top_clusters = pd.DataFrame(top_clusters).rename(columns={0:'hotel_cluster'})

	joblib.dump(top_clusters, dump_path)

print 'building srch_destination_id model...'
gen_top_k_group_by_model('srch_destination_id')
print 'building hotel_market model...'
gen_top_k_group_by_model('hotel_market')
print 'building orig_destination_distance model...'
gen_top_k_group_by_model('orig_destination_distance')
print 'building user_id model...'
gen_top_k_group_by_model('user_id')

print 'building srch_destination_id 2013 model...'
gen_top_k_group_by_model('srch_destination_id', year = '2013')
print 'building hotel_market 2013 model...'
gen_top_k_group_by_model('hotel_market', year = '2013')
print 'building orig_destination_distance 2013 model...'
gen_top_k_group_by_model('orig_destination_distance', year = '2013')
print 'building user_id 2013 model...'
gen_top_k_group_by_model('user_id', year = '2013')

print 'building srch_destination_id 2014 model...'
gen_top_k_group_by_model('srch_destination_id', year = '2014')
print 'building hotel_market 2014 model...'
gen_top_k_group_by_model('hotel_market', year = '2014')
print 'building orig_destination_distance 2014 model...'
gen_top_k_group_by_model('orig_destination_distance', year = '2014')
print 'building user_id 2014 model...'
gen_top_k_group_by_model('user_id', year = '2014')
