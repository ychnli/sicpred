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

problematic_member_id = ['r2i1231p1f1', 'r4i1231p1f1', 'r5i1231p1f1', 'r6i1231p1f1', 'r7i1231p1f1']

def load_config(config_path):
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

def check_for_sst_issue(config):
    if config.DATA_SPLIT_SETTINGS["member_ids"] == None:
        member_ids = config.DATA_SPLIT_SETTINGS["test"] + config.DATA_SPLIT_SETTINGS["val"] + config.DATA_SPLIT_SETTINGS["train"]
        if len(set(member_ids) & set(problematic_member_id)) != 0:
            if config.INPUT_CONFIG["sst"]["include"]:
                raise ValueError("this data split contains ensemble members with corrupted SST")

def main():
    parser = argparse.ArgumentParser(description="prepare data with specified config")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file (e.g., config.py)")
    parser.add_argument("--overwrite", action="store_true", help="If set, overwrite existing files.")

    args = parser.parse_args()

    # load the config variables
    config = load_config(args.config)
    check_for_sst_issue(config) 

    # create directories for saving processed data
    os.makedirs(os.path.join(config_cesm.PROCESSED_DATA_DIRECTORY, "normalized_inputs", config.DATA_CONFIG_NAME), exist_ok=True)
    os.makedirs(os.path.join(config_cesm.PROCESSED_DATA_DIRECTORY, "data_pairs", config.DATA_CONFIG_NAME), exist_ok=True)

    # merge downloaded data, if not already
    util_cesm.merge_data_by_member()

    # Normalize 
    print("Normalizing data according to the following data_split_settings:")
    pprint.pprint(config.DATA_SPLIT_SETTINGS, sort_dicts=False)
    print("\n")

    for var_name in config.INPUT_CONFIG.keys():
        if config.INPUT_CONFIG[var_name]['include'] and config.INPUT_CONFIG[var_name]['norm']:
            divide_by_stdev = config.INPUT_CONFIG[var_name]['divide_by_stdev']
            use_min_max = config.INPUT_CONFIG[var_name]['use_min_max']
            util_cesm.normalize_data(var_name, config.DATA_SPLIT_SETTINGS,
                                    max_lag_months=config.INPUT_CONFIG[var_name]["lag"],
                                    max_lead_months=config.MAX_LEAD_MONTHS,
                                    overwrite=args.overwrite, verbose=2, divide_by_stdev=divide_by_stdev, 
                                    use_min_max=use_min_max)

    print("done! \n\n")

    # save the SST land mask
    util_cesm.save_land_mask() 

    # save the icefrac land mask
    util_cesm.save_icefrac_land_mask() 

    # compute month weights
    print("Calculating and saving month weights... \n")
    month_weights_fp = os.path.join(config_cesm.PROCESSED_DATA_DIRECTORY, "normalized_inputs", config.DATA_CONFIG_NAME, "month_weights.pkl")
    if not os.path.exists(month_weights_fp) or args.overwrite:
        month_weights = util_cesm.calculate_monthly_weights(data_split_settings=config.DATA_SPLIT_SETTINGS)
        with open(month_weights_fp, "wb") as f:
            pickle.dump(month_weights, f)
        print("done! \n\n")

    # Prepare model-ready data pairs (concatenate stuff) 
    print("Prepping model-ready data pairs... \n")
    model_data_save_path = os.path.join(config_cesm.PROCESSED_DATA_DIRECTORY, "data_pairs", config.DATA_CONFIG_NAME)
    os.makedirs(model_data_save_path, exist_ok=True)

    util_cesm.save_inputs_files(config.INPUT_CONFIG, model_data_save_path, config.DATA_SPLIT_SETTINGS, overwrite=args.overwrite)

    util_cesm.save_targets_files(config.INPUT_CONFIG, config.TARGET_CONFIG, model_data_save_path,
                                 config.MAX_LEAD_MONTHS, config.DATA_SPLIT_SETTINGS, overwrite=args.overwrite)
    print("all done! \n\n")


if __name__ == "__main__":
    main()