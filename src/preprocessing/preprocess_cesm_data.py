######################################################################
# This script normalizes the CESM data and then concatenates the data
# into model-ready data pairs and saves them. 
#
######################################################################

import os 
import pprint
import argparse
import importlib
import pickle
from src.utils import util_cesm
from src import config_cesm

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

    # create directories for saving processed data
    os.makedirs(os.path.join(config_cesm.PROCESSED_DATA_DIRECTORY, "normalized_inputs", config.DATA_CONFIG_NAME), exist_ok=True)
    os.makedirs(os.path.join(config_cesm.PROCESSED_DATA_DIRECTORY, "data_pairs", config.DATA_CONFIG_NAME), exist_ok=True)

    # Normalize 
    print("Normalizing data according to the following data_split_settings:")
    pprint.pprint(config.DATA_SPLIT_SETTINGS, sort_dicts=False)
    print("\n")

    for var_name in config.INPUT_CONFIG.keys():
        if config.INPUT_CONFIG[var_name]['include'] and config.INPUT_CONFIG[var_name]['norm']:
            divide_by_stdev = config.INPUT_CONFIG[var_name]['divide_by_stdev']
            util_cesm.normalize_data(var_name, config.DATA_SPLIT_SETTINGS,
                                    max_lag_months=config.INPUT_CONFIG[var_name]["lag"],
                                    max_lead_months=config.MAX_LEAD_MONTHS,
                                    overwrite=False, verbose=2, divide_by_stdev=divide_by_stdev)

    print("done! \n\n")

    # Prepare model-ready data pairs (concatenate stuff) 
    print("Prepping model-ready data pairs... \n")
    model_data_save_path = os.path.join(config_cesm.PROCESSED_DATA_DIRECTORY, "data_pairs", config.DATA_CONFIG_NAME)
    os.makedirs(model_data_save_path, exist_ok=True)

    util_cesm.save_inputs_files(config.INPUT_CONFIG, model_data_save_path, config.DATA_SPLIT_SETTINGS)

    util_cesm.save_targets_files(config.INPUT_CONFIG, config.TARGET_CONFIG, model_data_save_path,
                                 config.MAX_LEAD_MONTHS, config.DATA_SPLIT_SETTINGS)
    print("all done! \n\n")

    # keep track of which data settings have already been preprocessed
    # save_path = os.path.join(config_cesm.PROCESSED_DATA_DIRECTORY, "processed_data_configs.pkl")
    # if os.path.exists(save_path):
    #     with open(save_path, "rb") as f:
    #         processed_data_configs = pickle.load(f)
    #         processed_data_configs.append(config.DATA_SPLIT_SETTINGS) 
    # else: 
    #     processed_data_configs = []
    #     processed_data_configs.append(config.DATA_SPLIT_SETTINGS)
    
    # with open(save_path, "wb") as f:
    #     pickle.dump(f, processed_data_configs)

if __name__ == "__main__":
    main()