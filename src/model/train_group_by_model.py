import numpy as np
import pandas as pd
from sklearn.externals import joblib

import sys
import os
scriptpath = os.path.dirname(os.path.realpath(sys.argv[0])) + '/../'
sys.path.append(os.path.abspath(scriptpath))
import utils
from utils import *


train = pd.read_csv(utils.raw_data_path + 'train.csv',
                 	dtype={is_booking':bool,'srch_destination_id':np.int32, 'hotel_market':np.int32, \
                 	'orig_destination_distance':np.double, 'user_id':np.int32, 'hotel_cluster':np.int32}',
                    usecols=['data_time' , 'is_booking', 'srch_destination_id', 'hotel_market', \
                    'orig_destination_distance', 'user_id', 'hotel_cluster'])


train_2013 = train[train.date_time < '2014-01-01 00:00:00']
train_2014 = train[train.date_time >= '2014-01-01 00:00:00']

def top_k_relevence(group, k = utils.k):
    idx = group.relevance.nlargest(k).index
    top_k_relevence = group.hotel_cluster[idx].values
    return np.array_str(top_k_relevence)[1:-1] # remove square brackets

def gen_top_k_group_by_model(group_by_field, click_weight = utils.click_weight, year = 'all'):
	source = train
	if year == '2013':
		source = train_2013
	elseif year == '2014':
		source = train_2014
	
	agg = source.groupby([group_by_field, 'hotel_cluster'])['is_booking'].agg(['sum','count'])
	agg['count'] -= agg['sum']
	agg = agg.rename(columns={'sum':'bookings','count':'clicks'})
	agg['relevance'] = agg['bookings'] + click_weight * agg['clicks']
	agg.reset_index(inplace = True)
	top_clusters = agg.groupby([group_by_field]).apply(top_k_relevence)
	top_clusters = pd.DataFrame(top_clusters).rename(columns={0:'hotel_cluster'})

	joblib.dump(top_clusters, utils.model_path + 
		'_'.join(['top', str(utils.k), 'cw', str(utils.click_weight), 'group', group_by_field, 'year', year]) + '.pkl')

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
