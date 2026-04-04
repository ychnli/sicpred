###################################################################################
# This script evaluates models on the observational dataset for the finetuning experiment.
#
# Results are saved to PREDICTIONS_DIRECTORY which should be set in config_cesm.py
###################################################################################

python3 -m src.models.evaluate --config src/experiment_configs/exp3_obs/obs_input2.py
python3 -m src.models.evaluate --config src/experiment_configs/exp3_obs/obs_input2_finetune.py