###################################################################################
# This script runs the diagnostics and bootstrap confidence intervals for the 
# variable importance experiment (exp1)
# 
# Bootstrap results are saved in ANALYSIS_RESULTS_DIRECTORY which is set in config_cesm.py
###################################################################################

python -m src.models.diagnostics --config exp1_inputs:input3e
python -m src.models.diagnostics --config exp1_inputs:input3f
python -m src.models.diagnostics --config exp1_inputs:input4a

# compute bootstrap confidence intervals
python -m src.utils.bootstrap --metric acc --config_a exp1_input2 --config_b exp1_input3e --transform fisher_z 
python -m src.utils.bootstrap --metric acc --config_a exp1_input2 --config_b exp1_input3f --transform fisher_z 
python -m src.utils.bootstrap --metric acc --config_a exp1_input2 --config_b exp1_input4a --transform fisher_z 

python -m src.utils.bootstrap --metric rmse --config_a exp1_input2 --config_b exp1_input3e 
python -m src.utils.bootstrap --metric rmse --config_a exp1_input2 --config_b exp1_input3f 
python -m src.utils.bootstrap --metric rmse --config_a exp1_input2 --config_b exp1_input4a 
