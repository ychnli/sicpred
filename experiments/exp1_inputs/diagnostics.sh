python -m src.models.diagnostics --config src/experiment_configs/exp1_inputs/input2.py 
python -m src.models.diagnostics --config src/experiment_configs/exp1_inputs/input3a.py 
python -m src.models.diagnostics --config src/experiment_configs/exp1_inputs/input3b.py 
python -m src.models.diagnostics --config src/experiment_configs/exp1_inputs/input3c.py 
python -m src.models.diagnostics --config src/experiment_configs/exp1_inputs/input3d.py 
python -m src.models.diagnostics --config src/experiment_configs/exp1_inputs/input4.py

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