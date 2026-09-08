"""Normalize CESM inputs and construct model-ready input-target pairs.

Command-line usage:
    --config SELECTOR          Required experiment configuration selector.
    --overwrite                Regenerate existing normalized artifacts instead of skipping them.

Example:
    python -m src.preprocessing.preprocess_cesm_data --config exp1_inputs:input3e
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
    data_pairs_dir = os.path.join(
        config_cesm.PROCESSED_DATA_DIRECTORY, "data_pairs", config.data_name
    )
    os.makedirs(data_pairs_dir, exist_ok=True)

    # merge downloaded data, if not already
    util_cesm.merge_data_by_member()

    # Normalize 
    print("Normalizing data according to the following data_split_settings:")
    pprint.pprint(config.data_split, sort_dicts=False)
    print("\n")

    for var_name in config.input_config.keys():
        settings = config.input_config[var_name]
        # icefrac is also the prediction target, so its normalized data is
        # required even when it is intentionally excluded as a predictor.
        if (settings['include'] or var_name == "icefrac") and settings['norm']:
            divide_by_stdev = settings['divide_by_stdev']
            use_min_max = settings['use_min_max']
            util_cesm.normalize_data(var_name, config.data_split,
                                    max_lag_months=settings["lag"],
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

    print("Constructing model-ready input-target pairs... \n")
    util_cesm.save_inputs_files(
        config.input_config, data_pairs_dir, config.data_split,
        overwrite=args.overwrite,
    )
    util_cesm.save_targets_files(
        config.target_config, data_pairs_dir, config.max_lead_months,
        config.data_split, overwrite=args.overwrite,
    )

    print("all done! \n\n")


if __name__ == "__main__":
    main()