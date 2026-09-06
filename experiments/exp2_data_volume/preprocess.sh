###################################################################################
# This script runs the data preprocessing for the data scaling experiment (exp2)
# 
# Results are saved in PROCESSED_DATA_DIRECTORY, which is set in config_cesm.py
###################################################################################

python3 -m src.preprocessing.preprocess_cesm_data --config exp2_data_volume:vol1
python3 -m src.preprocessing.preprocess_cesm_data --config exp2_data_volume:vol2
python3 -m src.preprocessing.preprocess_cesm_data --config exp2_data_volume:vol3
python3 -m src.preprocessing.preprocess_cesm_data --config exp2_data_volume:vol4
