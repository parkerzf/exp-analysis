for year in 2013 2014 all
do
    for clicks_weight in 0.01 0.02 0.04 0.06 0.08 0.1 0.2
    do
        echo python utils.py -set_params 5 ${clicks_weight} ${year}
        python utils.py -set_params 5 ${clicks_weight} ${year}
    	
        echo python model/train_group_by_model.py
    	python model/train_group_by_model.py
    	
        echo python model/predict_group_by_model.py
    	python model/predict_group_by_model.py
    done
done