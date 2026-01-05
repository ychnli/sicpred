python3 -m src.models.train --config src/experiment_configs/exp3_obs/obs_input2_finetune.py --pretrained /scratch/users/yucli/sicpred_models/exp2_vol4/UNetRes3_exp2_vol4_member_0_best.pth --members 20
python3 -m src.models.evaluate --config src/experiment_configs/exp3_obs/obs_input2_finetune.py --overwrite

python3 -m src.models.train --config src/experiment_configs/exp3_obs/obs_input4_finetune.py --pretrained /scratch/users/yucli/sicpred_models/exp1_input4/UNetRes3_exp1_input4_member_0_best.pth --members 20
python3 -m src.models.evaluate --config src/experiment_configs/exp3_obs/obs_input4_finetune.py

# copied from diagnostics
python3 -m src.models.diagnostics --config src/experiment_configs/exp3_obs/obs_input2.py --overwrite
python3 -m src.models.diagnostics --config src/experiment_configs/exp3_obs/obs_input4.py --overwrite
python3 -m src.models.diagnostics --config src/experiment_configs/exp3_obs/obs_input2_finetune.py
python3 -m src.models.diagnostics --config src/experiment_configs/exp3_obs/obs_input4_finetune.py

python3 -m src.utils.bootstrap --metric acc --config_a obs_input2_finetune --config_b obs_input2_ensemble --transform fisher_z --overwrite
python3 -m src.utils.bootstrap --metric acc --config_a obs_input4_finetune --config_b obs_input4_ensemble --transform fisher_z --overwrite
python3 -m src.utils.bootstrap --metric rmse --config_a obs_input2_finetune --config_b obs_input2_ensemble --overwrite
python3 -m src.utils.bootstrap --metric rmse --config_a obs_input4_finetune --config_b obs_input4_ensemble --overwrite
