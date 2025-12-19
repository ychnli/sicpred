python3 -m src.models.train --config src/experiment_configs/exp1_inputs/input2.py --members 1
python3 -m src.models.evaluate --config src/experiment_configs/exp1_inputs/input2.py

python3 -m src.models.train --config src/experiment_configs/exp1_inputs/input3a.py --members 1
python3 -m src.models.evaluate --config src/experiment_configs/exp1_inputs/input3a.py

python3 -m src.models.train --config src/experiment_configs/exp1_inputs/input3b.py --members 1
python3 -m src.models.evaluate --config src/experiment_configs/exp1_inputs/input3b.py