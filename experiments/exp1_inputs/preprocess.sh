###################################################################################
# This script runs the data preprocessing for the variable importance experiment 
# (exp1)
# 
# Results are saved in PROCESSED_DATA_DIRECTORY, which is set in config_cesm.py
###################################################################################

python3 -m src.preprocessing.preprocess_cesm_data --config exp1_inputs:input2
python3 -m src.preprocessing.preprocess_cesm_data --config exp1_inputs:input3a
python3 -m src.preprocessing.preprocess_cesm_data --config exp1_inputs:input3b
python3 -m src.preprocessing.preprocess_cesm_data --config exp1_inputs:input3c
python3 -m src.preprocessing.preprocess_cesm_data --config exp1_inputs:input3d
python3 -m src.preprocessing.preprocess_cesm_data --config exp1_inputs:input4