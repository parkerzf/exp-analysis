.PHONY: setting install analysis train predict clean all
#################################################################################
# COMMANDS                                                                      #
#################################################################################

# group by model
install:
	pip install -r requirements.txt

setting: install
	python src/utils.py -set_params 5 0.2 all

analysis: install
	python src/analysis/feature_analysis.py

train: setting
	python src/model/train_group_by_model.py

predict: setting
	python src/model/predict_group_by_model.py

clean:
	find . -name "*.pyc" -exec rm {} \;

group: install setting train predict

# rf model
setting_rf: install
	python src/utils.py -set_params 5 0.05 all

data_rf: setting_rf
	python src/features/build_baseline_features.py
	python src/features/build_group_by_features.py
	python src/features/concat_features.py

train_rf: setting_rf
	python src/model/train_rf_model.py

predict_rf: setting_rf
	python src/model/predict_rf_model.py

rf: install setting_rf data_rf train_rf predict_rf

# xgboost model
setting_xgb: install
	python src/utils.py -set_params 5 0.05 all

data_xgb: setting_xgb
	python src/features/build_baseline_features.py
	python src/features/build_group_by_features.py
	python src/features/concat_features.py

train_xgb: setting_xgb
	python src/model/train_xgb_model.py

predict_xgb: setting_xgb
	python src/model/predict_xgb_model.py

xgb: install setting_xgb data_xgb train_xgb predict_xgb






