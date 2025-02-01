
python3 -m src.preprocessing.preprocess_cesm_data --config src/experiment_configs/config_simple_plus_psl.py
python3 -m src.models.train --config src/experiment_configs/config_simple_plus_psl.py
python3 -m src.models.evaluate --config src/experiment_configs/config_simple_plus_psl.py

# python3 -m src.preprocessing.preprocess_cesm_data --config src/experiment_configs/config3.py
# python3 -m src.models.train --config src/experiment_configs/config3.py 