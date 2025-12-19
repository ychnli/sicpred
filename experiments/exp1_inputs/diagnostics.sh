<<<<<<< HEAD
python -m src.models.diagnostics --config src/experiment_configs/exp1_inputs/input2.py 
python -m src.models.diagnostics --config src/experiment_configs/exp1_inputs/input3a.py 
python -m src.models.diagnostics --config src/experiment_configs/exp1_inputs/input3b.py 
python -m src.models.diagnostics --config src/experiment_configs/exp1_inputs/input3c.py 
python -m src.models.diagnostics --config src/experiment_configs/exp1_inputs/input3d.py 
python -m src.models.diagnostics --config src/experiment_configs/exp1_inputs/input4.py

# compute bootstrap confidence intervals
python -m src.utils.bootstrap --metric acc --config_a exp1_input2 --config_b exp1_input3a --transform fisher_z --overwrite
python -m src.utils.bootstrap --metric acc --config_a exp1_input2 --config_b exp1_input3b --transform fisher_z --overwrite
python -m src.utils.bootstrap --metric acc --config_a exp1_input2 --config_b exp1_input3c --transform fisher_z --overwrite
python -m src.utils.bootstrap --metric acc --config_a exp1_input2 --config_b exp1_input3d --transform fisher_z --overwrite
python -m src.utils.bootstrap --metric acc --config_a exp1_input2 --config_b exp1_input4 --transform fisher_z --overwrite

python -m src.utils.bootstrap --metric rmse --config_a exp1_input2 --config_b exp1_input3a --overwrite
python -m src.utils.bootstrap --metric rmse --config_a exp1_input2 --config_b exp1_input3b --overwrite
python -m src.utils.bootstrap --metric rmse --config_a exp1_input2 --config_b exp1_input3c --overwrite
python -m src.utils.bootstrap --metric rmse --config_a exp1_input2 --config_b exp1_input3d --overwrite
python -m src.utils.bootstrap --metric rmse --config_a exp1_input2 --config_b exp1_input4 --overwrite
=======
python -m src.models.diagnostics --config src/experiment_configs/exp1_inputs/input2.py --overwrite
python -m src.models.diagnostics --config src/experiment_configs/exp1_inputs/input3a.py --overwrite
python -m src.models.diagnostics --config src/experiment_configs/exp1_inputs/input3b.py --overwrite
python -m src.models.diagnostics --config src/experiment_configs/exp1_inputs/input3c.py --overwrite
python -m src.models.diagnostics --config src/experiment_configs/exp1_inputs/input3d.py --overwrite
python -m src.models.diagnostics --config src/experiment_configs/exp1_inputs/input4.py --overwrite
>>>>>>> 0b3d20d (updaet ACC and RMSE figures in exp1)
