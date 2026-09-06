###################################################################################
# This script trains models of 6 different input configurations for the variable 
# importance experiment (exp1). 5 neural ensemble members are trained for each 
# configuration
# 
# Results are saved to MODEL_DIRECTORY which should be set in config_cesm.py
# 
# A GPU is recommended for this script
###################################################################################

python3 -m src.models.train --config exp1_inputs:input2 --members 5
python3 -m src.models.train --config exp1_inputs:input3a --members 5
python3 -m src.models.train --config exp1_inputs:input3b --members 5
python3 -m src.models.train --config exp1_inputs:input3c --members 5
python3 -m src.models.train --config exp1_inputs:input3d --members 5
python3 -m src.models.train --config exp1_inputs:input4 --members 5
