python -m src.models.diagnostics --config src/experiment_configs/exp2_data_volume/vol1.py 
python -m src.models.diagnostics --config src/experiment_configs/exp2_data_volume/vol2.py 
python -m src.models.diagnostics --config src/experiment_configs/exp2_data_volume/vol3.py 
python -m src.models.diagnostics --config src/experiment_configs/exp2_data_volume/vol4.py 

python -m src.utils.bootstrap --metric acc --config_a exp2_vol1 --config_b exp2_vol2 --transform fisher_z --overwrite
python -m src.utils.bootstrap --metric acc --config_a exp2_vol2 --config_b exp2_vol3 --transform fisher_z --overwrite
python -m src.utils.bootstrap --metric acc --config_a exp2_vol3 --config_b exp2_vol4 --transform fisher_z --overwrite