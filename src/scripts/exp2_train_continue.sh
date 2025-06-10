python3 -m src.models.train --config src/experiment_configs/exp2_data_volume/vol2.py --resume 10
python3 -m src.models.train --config src/experiment_configs/exp2_data_volume/vol3.py --resume 10
python3 -m src.models.train --config src/experiment_configs/exp2_data_volume/vol4.py --resume 15

# python3 -m src.models.evaluate --config src/experiment_configs/config_test_larger_dataset.py