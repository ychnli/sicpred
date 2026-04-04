###################################################################################
# This script evaluates models of 6 different input configurations for the variable 
# importance experiment (exp1). 5 neural ensemble members are evaluated for each 
# configuration
# 
# Results are saved to PREDICTIONS_DIRECTORY which should be set in config_cesm.py
###################################################################################

python3 -m src.models.evaluate --config src/experiment_configs/exp1_inputs/input2.py
python3 -m src.models.evaluate --config src/experiment_configs/exp1_inputs/input3a.py
python3 -m src.models.evaluate --config src/experiment_configs/exp1_inputs/input3b.py
python3 -m src.models.evaluate --config src/experiment_configs/exp1_inputs/input3c.py
python3 -m src.models.evaluate --config src/experiment_configs/exp1_inputs/input3d.py
python3 -m src.models.evaluate --config src/experiment_configs/exp1_inputs/input4.py
