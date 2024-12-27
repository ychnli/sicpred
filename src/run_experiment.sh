
#python3 -m src.preprocessing.preprocess_cesm_data --config src/experiment_configs/config2.py
python3 -m src.models.train --config src/experiment_configs/config2.py

python3 -m src.preprocessing.preprocess_cesm_data --config src/experiment_configs/config3.py
python3 -m src.models.train --config src/experiment_configs/config3.py 