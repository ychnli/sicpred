###################################################################################
# This script runs the data preprocessing for the finetuning experiment (exp3)
# 
# Results are saved in PROCESSED_DATA_DIRECTORY, which is set in config_cesm.py
###################################################################################

python3 -m src.preprocessing.preprocess_cesm_data --config exp3_obs:obs_input2

