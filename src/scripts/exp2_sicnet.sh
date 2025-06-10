python3 -m src.preprocessing.preprocess_cesm_data --config src/experiment_configs/exp2_data_volume/vol3_m2.py 
python3 -m src.models.train --config src/experiment_configs/exp2_data_volume/vol3_m2.py 
python3 -m src.models.evaluate --config src/experiment_configs/exp2_data_volume/vol3_m2.py 