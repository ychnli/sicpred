# preprocess data
python3 -m src.preprocessing.preprocess_cesm_data --config src/experiment_configs/exp1_inputs/input3.py
python3 -m src.preprocessing.preprocess_cesm_data --config src/experiment_configs/exp1_inputs/input4.py
python3 -m src.preprocessing.preprocess_cesm_data --config src/experiment_configs/exp1_inputs/input5.py

python3 -m src.models.train --config src/experiment_configs/exp1_inputs/input3.py --members 5
python3 -m src.models.train --config src/experiment_configs/exp1_inputs/input4.py --members 5


# python3 -m src.models.evaluate --config src/experiment_configs/config_test_larger_dataset.py