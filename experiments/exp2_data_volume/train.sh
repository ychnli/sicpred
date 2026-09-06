###################################################################################
# This script trains and evaluates models of 4 different data volume configurations 
# for the data scaling experiment (exp2). 
#
# Results are saved to MODEL_DIRECTORY which should be set in config_cesm.py
# 
# A GPU is recommended for this script. Note that most of the time will be spent 
# training the model with the most training data (vol3 and vol4). 
###################################################################################

python3 -m src.models.train --config exp2_data_volume:vol1 --members 5
python3 -m src.models.train --config exp2_data_volume:vol2 --members 3
python3 -m src.models.train --config exp2_data_volume:vol3 --members 1
python3 -m src.models.train --config exp2_data_volume:vol4 --members 1
