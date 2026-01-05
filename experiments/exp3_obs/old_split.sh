# try the old split
python3 -m src.preprocessing.preprocess_cesm_data --config src/experiment_configs/exp3_obs/obs_input2_oldsplit.py

python3 -m src.models.train --config src/experiment_configs/exp3_obs/obs_input2_oldsplit.py --members 5
python3 -m src.models.evaluate --config src/experiment_configs/exp3_obs/obs_input2_oldsplit.py

python3 -m src.models.train --config src/experiment_configs/exp3_obs/obs_input2_oldsplit_ft.py --pretrained /scratch/users/yucli/sicpred_models/exp2_vol4/UNetRes3_exp2_vol4_member_0_best.pth --members 5
python3 -m src.models.evaluate --config src/experiment_configs/exp3_obs/obs_input2_oldsplit_ft.py 

python3 -m src.models.diagnostics --config src/experiment_configs/exp3_obs/obs_input2_oldsplit.py
python3 -m src.models.diagnostics --config src/experiment_configs/exp3_obs/obs_input2_oldsplit_ft.py

