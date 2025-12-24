python3 -m src.models.train --config src/experiment_configs/exp2_data_volume/vol1.py --members 5
python3 -m src.models.evaluate --config src/experiment_configs/exp2_data_volume/vol1.py

python3 -m src.models.train --config src/experiment_configs/exp2_data_volume/vol2.py --members 3
python3 -m src.models.evaluate --config src/experiment_configs/exp2_data_volume/vol2.py