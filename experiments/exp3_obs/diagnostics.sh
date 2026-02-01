# evaluate and compute diagnostics for non-pretrained models
# python3 -m src.models.evaluate --config src/experiment_configs/exp3_obs/obs_input2.py --overwrite
python3 -m src.models.evaluate --config src/experiment_configs/exp3_obs/obs_input4.py --overwrite
# python3 -m src.models.diagnostics --config src/experiment_configs/exp3_obs/obs_input2.py --overwrite --ensemble-mean --baselines
python3 -m src.models.diagnostics --config src/experiment_configs/exp3_obs/obs_input4.py --overwrite --ensemble-mean

# evaluate and compute diagnostics for pretrained models
# python3 -m src.models.evaluate --config src/experiment_configs/exp3_obs/obs_input2_finetune.py --overwrite
python3 -m src.models.evaluate --config src/experiment_configs/exp3_obs/obs_input4_finetune.py --overwrite
# python3 -m src.models.diagnostics --config src/experiment_configs/exp3_obs/obs_input2_finetune.py --overwrite --ensemble-mean
python3 -m src.models.diagnostics --config src/experiment_configs/exp3_obs/obs_input4_finetune.py --overwrite --ensemble-mean

# compute bootstrap confidence intervals for where the diagnostics are different
# python -m src.utils.bootstrap --metric acc --config_a obs_input2_finetune --config_b obs_input2_ensemble --transform fisher_z --overwrite
python -m src.utils.bootstrap --metric acc --config_a obs_input4_finetune --config_b obs_input4_ensemble --transform fisher_z --overwrite
# python -m src.utils.bootstrap --metric rmse --config_a obs_input2_finetune --config_b obs_input2_ensemble --overwrite
python -m src.utils.bootstrap --metric rmse --config_a obs_input4_finetune --config_b obs_input4_ensemble --overwrite

# run zero-shot evaluations of the pretrained model
# python3 -m src.models.evaluate --config src/experiment_configs/exp2_data_volume/vol4.py --zero-shot src/experiment_configs/exp3_obs/obs_input2.py --overwrite
# python3 -m src.models.diagnostics --config src/experiment_configs/exp3_obs/obs_input2.py --predictions-path /scratch/users/yucli/sicpred_model_predictions/obs_input2_ensemble/exp2_vol4_UNetRes3_zeroshot_predictions.nc --label _cesm_zs --overwrite