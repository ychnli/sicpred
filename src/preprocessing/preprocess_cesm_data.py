######################################################################
# This script normalizes the CESM data and then concatenates the data
# into model-ready data pairs and saves them. 
#
# TODO: implement a way to specify how to normalize the data 
# and thereby run different experiments 
######################################################################

import os 
import argparse
import importlib
from src.utils import util_cesm


def load_config(config_path):
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def main():
    parser = argparse.ArgumentParser(description="prepare data with specified config")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file (e.g., config.py)")
    args = parser.parse_args()

    # load the config variables
    config = load_config(args.config)

    # Normalize 
    print("Normalizing data... \n")
    vars_to_normalize = config.VAR_NAMES
    for var_name in vars_to_normalize:
        if var_name in ["icefrac", "icethick"]:
            divide_by_stdev = False
        else: 
            divide_by_stdev = True
            
        util_cesm.normalize_data(var_name, normalization_scheme=None, 
                                overwrite=False, verbose=2, divide_by_stdev=divide_by_stdev)
    print("done! \n\n")

    # Prepare model-ready data pairs (concatenate stuff) 
    print("Prepping model-ready data pairs... \n")
    model_data_save_path = os.path.join(config_cesm.MODEL_DATA_DIRECTORY, "data_pairs_setting1")
    os.makedirs(model_data_save_path, exist_ok=True)
    util_cesm.save_inputs_files(config_cesm.input_config_all, model_data_save_path)
    util_cesm.save_targets_files(config_cesm.input_config_all, config_cesm.target_config, model_data_save_path)
    print("all done! \n\n")

if __name__ == "__main__":
    main()