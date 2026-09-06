###################################################################################
# This script evaluates models of 4 different data volume configurations 
# for the data scaling experiment (exp2). 
#
# Results are saved to PREDICTIONS_DIRECTORY which should be set in config_cesm.py
###################################################################################

python3 -m src.models.evaluate --config exp2_data_volume:vol1
python3 -m src.models.evaluate --config exp2_data_volume:vol2
python3 -m src.models.evaluate --config exp2_data_volume:vol3
python3 -m src.models.evaluate --config exp2_data_volume:vol4