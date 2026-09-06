###################################################################################
# This script evaluates models of 6 different input configurations for the variable 
# importance experiment (exp1). 5 neural ensemble members are evaluated for each 
# configuration
# 
# Results are saved to PREDICTIONS_DIRECTORY which should be set in config_cesm.py
###################################################################################

python3 -m src.models.evaluate --config exp1_inputs:input2
python3 -m src.models.evaluate --config exp1_inputs:input3a
python3 -m src.models.evaluate --config exp1_inputs:input3b
python3 -m src.models.evaluate --config exp1_inputs:input3c
python3 -m src.models.evaluate --config exp1_inputs:input3d
python3 -m src.models.evaluate --config exp1_inputs:input4
