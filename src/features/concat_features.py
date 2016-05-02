import numpy as np
import pandas as pd
import sys
import os
from sklearn.externals import joblib

scriptpath = os.path.dirname(os.path.realpath(sys.argv[0])) + '/../'
sys.path.append(os.path.abspath(scriptpath))
import utils

def dump(dataset_prefix):
	baseline = pd.read_csv(utils.processed_data_path + dataset_prefix + '_baseline_year_all.csv')
	group_by = pd.read_csv(utils.processed_data_path + dataset_prefix + '_group_by_top_5_cw_0.05_year_all.csv')

	data_all = pd.concat([baseline, group_by], axis = 1)

	joblib.dump(data_all, utils.processed_data_path + dataset_prefix + '_all_top_5_cw_0.05_year_all.pkl')


dump('train_is_booking')
dump('test')
