###################################################################################
# This script runs the diagnostics and bootstrap confidence intervals for the 
# variable importance experiment (exp1)
# 
# Results are saved in ANALYSIS_RESULTS_DIRECTORY which is set in config_cesm.py
###################################################################################

python -m src.models.diagnostics --config exp1_inputs:input2
python -m src.models.diagnostics --config exp1_inputs:input3a
python -m src.models.diagnostics --config exp1_inputs:input3b
python -m src.models.diagnostics --config exp1_inputs:input3c
python -m src.models.diagnostics --config exp1_inputs:input3d
python -m src.models.diagnostics --config exp1_inputs:input4

# compute bootstrap confidence intervals
python -m src.utils.bootstrap --metric acc --config_a exp1_input2 --config_b exp1_input3a --transform fisher_z 
python -m src.utils.bootstrap --metric acc --config_a exp1_input2 --config_b exp1_input3b --transform fisher_z 
python -m src.utils.bootstrap --metric acc --config_a exp1_input2 --config_b exp1_input3c --transform fisher_z 
python -m src.utils.bootstrap --metric acc --config_a exp1_input2 --config_b exp1_input3d --transform fisher_z 
python -m src.utils.bootstrap --metric acc --config_a exp1_input2 --config_b exp1_input4 --transform fisher_z 

python -m src.utils.bootstrap --metric rmse --config_a exp1_input2 --config_b exp1_input3a 
python -m src.utils.bootstrap --metric rmse --config_a exp1_input2 --config_b exp1_input3b 
python -m src.utils.bootstrap --metric rmse --config_a exp1_input2 --config_b exp1_input3c 
python -m src.utils.bootstrap --metric rmse --config_a exp1_input2 --config_b exp1_input3d 
python -m src.utils.bootstrap --metric rmse --config_a exp1_input2 --config_b exp1_input4 
