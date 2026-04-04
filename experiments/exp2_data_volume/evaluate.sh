###################################################################################
# This script evaluates models of 4 different data volume configurations 
# for the data scaling experiment (exp2). 
#
# Results are saved to PREDICTIONS_DIRECTORY which should be set in config_cesm.py
###################################################################################

python3 -m src.models.evaluate --config src/experiment_configs/exp2_data_volume/vol1.py
python3 -m src.models.evaluate --config src/experiment_configs/exp2_data_volume/vol2.py
python3 -m src.models.evaluate --config src/experiment_configs/exp2_data_volume/vol3.py
python3 -m src.models.evaluate --config src/experiment_configs/exp2_data_volume/vol4.py