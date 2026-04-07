###################################################################################
# This script runs the diagnostics and bootstrap confidence intervals for the 
# finetuning experiment (exp3) 
#
# Results are saved in ANALYSIS_RESULTS_DIRECTORY which is set in config_cesm.py
###################################################################################

# compute diagnostics for pretrained and non-pretrained models
python3 -m src.models.diagnostics --config src/experiment_configs/exp3_obs/obs_input2.py --ensemble-mean --baselines
python3 -m src.models.diagnostics --config src/experiment_configs/exp3_obs/obs_input2_finetune.py --ensemble-mean

# compute bootstrap confidence intervals for where the diagnostics are different
python -m src.utils.bootstrap --metric acc --config_a obs_input2_finetune --config_b obs_input2_ensemble --transform fisher_z
python -m src.utils.bootstrap --metric rmse --config_a obs_input2_finetune --config_b obs_input2_ensemble

# run zero-shot evaluations of the pretrained model
# first, get the location of the model predictions by extracting it from config_cesm.py
PREDICTIONS_DIRECTORY=$(python3 - <<EOF
from src.config_cesm import PREDICTIONS_DIRECTORY
print(PREDICTIONS_DIRECTORY)
EOF
)
PREDICTIONS_PATH="$PREDICTIONS_DIRECTORY/obs_input2_ensemble/exp2_vol4_UNetRes3_zeroshot_predictions.nc"

python3 -m src.models.evaluate --config src/experiment_configs/exp2_data_volume/vol4.py --zero-shot src/experiment_configs/exp3_obs/obs_input2.py
python3 -m src.models.diagnostics --config src/experiment_configs/exp3_obs/obs_input2.py --predictions-path "$PREDICTIONS_PATH" --label _cesm_zs