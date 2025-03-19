
python3 -m src.preprocessing.preprocess_cesm_data --config src/experiment_configs/config_test_larger_dataset.py
python3 -m src.models.train --config src/experiment_configs/config_test_larger_dataset.py
python3 -m src.models.evaluate --config src/experiment_configs/config_test_larger_dataset.py