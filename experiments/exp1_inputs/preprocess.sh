###################################################################################
# This script runs the data preprocessing for the variable importance experiment 
# (exp1)
# 
# Results are saved in PROCESSED_DATA_DIRECTORY, which is set in config_cesm.py
###################################################################################

python3 -m src.preprocessing.preprocess_cesm_data --config src/experiment_configs/exp1_inputs/input2.py
python3 -m src.preprocessing.preprocess_cesm_data --config src/experiment_configs/exp1_inputs/input3a.py
python3 -m src.preprocessing.preprocess_cesm_data --config src/experiment_configs/exp1_inputs/input3b.py
python3 -m src.preprocessing.preprocess_cesm_data --config src/experiment_configs/exp1_inputs/input3c.py
python3 -m src.preprocessing.preprocess_cesm_data --config src/experiment_configs/exp1_inputs/input3d.py
python3 -m src.preprocessing.preprocess_cesm_data --config src/experiment_configs/exp1_inputs/input4.py