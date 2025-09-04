# preprocess the data
# python3 -m src.preprocessing.preprocess_cesm_data --config src/experiment_configs/exp1_inputs/input3a_dev.py --overwrite
# python3 -m src.preprocessing.preprocess_cesm_data --config src/experiment_configs/exp1_inputs/input3a_std.py --overwrite

python3 -m src.models.train --config src/experiment_configs/exp1_inputs/input3a_dev.py --members 3
python3 -m src.models.train --config src/experiment_configs/exp1_inputs/input3a_std.py --members 3
python3 -m src.models.train --config src/experiment_configs/exp1_inputs/input_noise.py --members 3

python3 -m src.models.evaluate --config src/experiment_configs/exp1_inputs/input3a_dev.py
python3 -m src.models.evaluate --config src/experiment_configs/exp1_inputs/input3a_std.py
python3 -m src.models.evaluate --config src/experiment_configs/exp1_inputs/input_noise.py
