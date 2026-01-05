python3 -m src.models.diagnostics --config src/experiment_configs/exp3_obs/obs_input2.py --overwrite
python3 -m src.models.diagnostics --config src/experiment_configs/exp3_obs/obs_input4.py --overwrite

python3 -m src.models.diagnostics --config src/experiment_configs/exp3_obs/obs_input2_finetune.py
python3 -m src.models.diagnostics --config src/experiment_configs/exp3_obs/obs_input4_finetune.py

python -m src.utils.bootstrap --metric acc --config_a obs_input2_finetune --config_b obs_input2_ensemble --transform fisher_z --overwrite
python -m src.utils.bootstrap --metric acc --config_a obs_input4_finetune --config_b obs_input4_ensemble --transform fisher_z --overwrite
python -m src.utils.bootstrap --metric rmse --config_a obs_input2_finetune --config_b obs_input2_ensemble --overwrite
python -m src.utils.bootstrap --metric rmse --config_a obs_input4_finetune --config_b obs_input4_ensemble --overwrite