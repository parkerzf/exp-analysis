import numpy as np
import pandas as pd
import sys
import os
from sklearn.externals import joblib

# where we can find training, test, and sampleSubmission.csv
raw_data_path = '/Users/zhaofeng/Project/expedia/data/raw/'
processed_data_path = '/Users/zhaofeng/Project/expedia/data/processed/'
# where we store model and results
model_path = '/Users/zhaofeng/Project/expedia/models/'


#############################################################
#################### model parameters    ####################
#############################################################
k = 5 # top k hotel clusters
click_weight = 0.05 # click weight, assuming the booking weight = 1
train_year = 'all' # training with all the data, 2013 only or 2014 only

try:
    params = joblib.load(model_path + 'params.pkl')
    k = params['k']
    click_weight = params['click_weight']
    train_year = params['train_year']
except:
    pass

def print_help():
    print "  Usage: python utils.py -set_params [k] [click_weight] [train_year=2013|2014|all]"
    print "Example: python utils.py -set_params 5 0.05 all"

def main():
    if len(sys.argv) == 5 and sys.argv[1] == '-set_params':
        try:
            k = int(sys.argv[2])
            click_weight =  float(sys.argv[3])
            train_year = sys.argv[4]
            assert train_year == '2013' or train_year == '2014' or train_year == 'all', \
            'train_year not in (2013|2014|all)'
            joblib.dump({'k': k, 'click_weight': click_weight, 'train_year': train_year}, model_path + 'params.pkl')
        except:
            print_help()
    else:
        print_help()

#############################################################
####################   util functions    ####################
#############################################################
def fill_na(dataset, prev_result, group_by_field):
    """
    Use another group by model to enrich the previous model result for empty result instances
    :param dataset: train/test dataset
    :param prev_result: the dataframe of the previous model result
    :param group_by_field: group by field to get the another group by model
    :return: the dataframe with the updated model result
    """
    group_by_model = joblib.load(model_path + \
        '_'.join(['top', str(k), 'cw', str(click_weight), 'group', group_by_field, 'year', train_year]) + '.pkl')
    merged_df = dataset.merge(group_by_model, how='left',left_on= group_by_field, right_index=True)
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


def fill_all_top_5(dataset, prev_result, group_by_field, type = 'test'):
    """
    Use another group by model to enrich the previous model result for result instances with less than 5 hotel clusters
    :param dataset: train/test dataset
    :param prev_result: the dataframe of the previous model result
    :param group_by_field: group by field to get the another group by model
    :return: the dataframe with the updated model result
    """
    group_by_model = joblib.load(model_path + \
        '_'.join(['top', str(k), 'cw', str(click_weight), 'group', group_by_field, 'year', train_year]) + '.pkl')
        # '_'.join(['top', str(k), 'cw', str(0.05), 'group', group_by_field, 'year', train_year]) + '.pkl')
    merged_df = dataset.merge(group_by_model, how='left',left_on= group_by_field, right_index=True)

    if type == 'train':
        merged_df.reset_index(inplace = True)
        merged_df = merged_df[['index', 'hotel_cluster_y']]
        merged_df.columns = ['id', 'hotel_cluster']
        prev_result.reset_index(inplace = True)
        prev_result = prev_result[['index', 'hotel_cluster']]
        prev_result.columns = ['id', 'hotel_cluster']

    prev_result['new_hotel_cluster'] = prev_result.apply(lambda row: fill_top_5(row, merged_df), axis=1)
    

    new_result = prev_result.drop('hotel_cluster', 1)
    if type == 'test':
        new_result.columns = ['id', 'hotel_cluster']
    elif type == 'train':
        new_result.columns = ['date_time', 'hotel_cluster']

    return new_result

def load_train(type = 'group_by'):
    if type == 'group_by':
        try:
            train = joblib.load(processed_data_path + 'train_group_by.pkl')
        except:
            print 'load raw train data'
            train = pd.read_csv(raw_data_path + 'train.csv',
                            dtype={'date_time': str, 'is_booking':np.int8,'srch_destination_id':np.int32, 'hotel_market':np.int32, \
                            'orig_destination_distance':np.double, 'user_id':np.int32, 'hotel_cluster':np.int32},
                            usecols=['date_time', 'is_booking', 'srch_destination_id', 'hotel_market', \
                            'orig_destination_distance', 'user_id', 'hotel_cluster'])
            joblib.dump(train, processed_data_path + 'train_group_by.pkl')
    elif type == 'baseline':
        try:
            train = joblib.load(processed_data_path + 'train_is_booking_baseline.pkl')
        except:
            print 'load raw train data'
            train = pd.read_csv(raw_data_path + 'train.csv',
                            dtype={'date_time':str, 'is_booking':np.int8, 'site_name':np.int32,'posa_continent':np.int32, 'user_location_country':np.int32, \
                            'user_location_region':np.int32, 'orig_destination_distance':np.double, \
                            'is_mobile':np.int8, 'is_package':np.int8, 'channel':np.int32, 'srch_ci':str, 'srch_co':str, \
                            'srch_adults_cnt':np.int32, 'srch_children_cnt':np.int32, 'srch_rm_cnt':np.int32, \
                            'srch_destination_type_id':np.int32, 'hotel_continent':np.int32, 'hotel_country':np.int32, \
                            'hotel_cluster':np.int32},
                            usecols=['date_time', 'is_booking', 'site_name', 'posa_continent', 'user_location_country', \
                            'user_location_region', 'orig_destination_distance', \
                            'is_mobile', 'is_package', 'channel', 'srch_ci', 'srch_co', \
                            'srch_adults_cnt', 'srch_children_cnt', 'srch_rm_cnt', \
                            'srch_destination_type_id', 'hotel_continent', 'hotel_country', \
                            'hotel_cluster'])

            train_is_booking = train[train.is_booking == 1]
            joblib.dump(train_is_booking, processed_data_path + 'train_is_booking_baseline.pkl')
    return train

def load_test(type = 'group_by'):
    if type == 'group_by':
        try:
            test = joblib.load(processed_data_path + 'test_group_by.pkl')
        except:
            print 'load raw test data'
            test = pd.read_csv(raw_data_path + 'test.csv',
                            dtype={'srch_destination_id':np.int32, 'hotel_market':np.int32, \
                            'orig_destination_distance':np.double, 'user_id':np.int32},
                            usecols=['srch_destination_id', 'hotel_market', \
                            'orig_destination_distance', 'user_id'])
            joblib.dump(test, processed_data_path + 'test_group_by.pkl')
    elif type == 'baseline':
        try:
            test = joblib.load(processed_data_path + 'test_baseline.pkl')
        except:
            print 'load raw test data'
            test = pd.read_csv(raw_data_path + 'test.csv',
                            dtype={'date_time':str, 'site_name':np.int32,'posa_continent':np.int32, 'user_location_country':np.int32, \
                            'user_location_region':np.int32, 'orig_destination_distance':np.double, \
                            'is_mobile':np.int8, 'is_package':np.int8, 'channel':np.int32, 'srch_ci':str, 'srch_co':str, \
                            'srch_adults_cnt':np.int32, 'srch_children_cnt':np.int32, 'srch_rm_cnt':np.int32, \
                            'srch_destination_type_id':np.int32, 'hotel_continent':np.int32, 'hotel_country':np.int32},
                            usecols=['date_time', 'site_name', 'posa_continent', 'user_location_country', \
                            'user_location_region', 'orig_destination_distance', \
                            'is_mobile', 'is_package', 'channel', 'srch_ci', 'srch_co', \
                            'srch_adults_cnt', 'srch_children_cnt', 'srch_rm_cnt', \
                            'srch_destination_type_id', 'hotel_continent', 'hotel_country'])

            joblib.dump(test, processed_data_path + 'test_baseline.pkl')
    return test


if __name__ == "__main__":
    main()