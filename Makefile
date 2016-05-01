.PHONY: setting install analysis train predict clean all
#################################################################################
# COMMANDS                                                                      #
#################################################################################

install:
	pip install -r requirements.txt

setting: install
	python src/utils.py -set_params 5 0.05 all

setting_2014: install
	python src/utils.py -set_params 5 0.05 2014

analysis: install
	python src/analysis/feature_analysis.py

train: install setting
	python src/model/train_group_by_model.py

predict: install setting
	python src/model/predict_group_by_model.py

clean:
	find . -name "*.pyc" -exec rm {} \;

all: install setting train predict

all_2014: install setting_2014 train predict



