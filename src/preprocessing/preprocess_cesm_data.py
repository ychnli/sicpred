"""
This script normalizes CESM data according to the data split settings
given by the desired configuration. 
"""

import os 
import pprint
import argparse
import pickle
from src.utils import util_cesm
from src import config_cesm
from src.experiment_configs import load_config

problematic_member_id = ['r2i1231p1f1', 'r4i1231p1f1', 'r5i1231p1f1', 'r6i1231p1f1', 'r7i1231p1f1']
def check_for_sst_issue(config):
    if config.data_split["member_ids"] == None:
        member_ids = config.data_split["test"] + config.data_split["val"] + config.data_split["train"]
        if len(set(member_ids) & set(problematic_member_id)) != 0:
            if config.input_config["sst"]["include"]:
                raise ValueError("this data split contains ensemble members with corrupted SST")

def main():
    parser = argparse.ArgumentParser(description="prepare data with specified config")
    parser.add_argument("--config", type=str, required=True, help="Named configuration selector (e.g., exp1_inputs:input2)")
    parser.add_argument("--overwrite", action="store_true", help="If set, overwrite existing files.")

    args = parser.parse_args()

    # load the config variables
    config = load_config(args.config)
    check_for_sst_issue(config) 

    # create directories for saving processed data
    os.makedirs(os.path.join(config_cesm.PROCESSED_DATA_DIRECTORY, "normalized_inputs", config.data_name), exist_ok=True)

    # merge downloaded data, if not already
    util_cesm.merge_data_by_member()

    # Normalize 
    print("Normalizing data according to the following data_split_settings:")
    pprint.pprint(config.data_split, sort_dicts=False)
    print("\n")

    for var_name in config.input_config.keys():
        if config.input_config[var_name]['include'] and config.input_config[var_name]['norm']:
            divide_by_stdev = config.input_config[var_name]['divide_by_stdev']
            use_min_max = config.input_config[var_name]['use_min_max']
            util_cesm.normalize_data(var_name, config.data_split,
                                    max_lag_months=config.input_config[var_name]["lag"],
                                    max_lead_months=config.max_lead_months,
                                    overwrite=args.overwrite, verbose=2, divide_by_stdev=divide_by_stdev, 
                                    use_min_max=use_min_max)

    print("done! \n\n")

    # save the SST land mask
    util_cesm.save_land_mask() 

    # save the icefrac land mask
    util_cesm.save_icefrac_land_mask() 

    # compute month weights
    print("Calculating and saving month weights... \n")
    month_weights_fp = os.path.join(config_cesm.PROCESSED_DATA_DIRECTORY, "normalized_inputs", config.data_name, "month_weights.pkl")
    if not os.path.exists(month_weights_fp) or args.overwrite:
        month_weights = util_cesm.calculate_monthly_weights(data_split_settings=config.data_split)
        with open(month_weights_fp, "wb") as f:
            pickle.dump(month_weights, f)
        print("done! \n\n")

    print("all done! \n\n")


if __name__ == "__main__":
    main()