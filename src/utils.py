import sys
from sklearn.externals import joblib

# where we can find training, test, and sampleSubmission.csv
raw_data_path = '/Users/zhaofeng/Project/expedia/data/raw/'
# where we store model and results
model_path = '/Users/zhaofeng/Project/expedia/models/'


#############################################################
#################### model parameters    ####################
#############################################################
k = 5
click_weight = 0.05
train_year = 'all'

try:
    params = joblib.load(model_path + 'params.pkl')
    k = params['k']
    click_weight = params['click_weight']
except:
    pass

def print_help():
    print "usage: python utils -set_params [k], [click_weight] [train_year=2013|2014|all]"
    print "for example: python utils -set_params 5 0.05 all"

def main():
    if sys.argv[1] == '-set_params' and len(sys.argv) == 5:
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

if __name__ == "__main__":
    main()