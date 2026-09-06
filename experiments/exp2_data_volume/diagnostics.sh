###################################################################################
# This script runs the diagnostics and bootstrap confidence intervals for the 
# data scaling experiment (exp2) 
#
# Results are saved in ANALYSIS_RESULTS_DIRECTORY which is set in config_cesm.py
###################################################################################

python -m src.models.diagnostics --config exp2_data_volume:vol1
python -m src.models.diagnostics --config exp2_data_volume:vol2
python -m src.models.diagnostics --config exp2_data_volume:vol3
python -m src.models.diagnostics --config exp2_data_volume:vol4 --baselines

python -m src.utils.bootstrap --metric acc --config_a exp2_vol1 --config_b exp2_vol2 --transform fisher_z
python -m src.utils.bootstrap --metric acc --config_a exp2_vol2 --config_b exp2_vol3 --transform fisher_z
python -m src.utils.bootstrap --metric acc --config_a exp2_vol3 --config_b exp2_vol4 --transform fisher_z