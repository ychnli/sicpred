"""Evaluate saved checkpoints after permuting one precomputed test-set input channel.

Command-line usage:
    --config SELECTOR          Required experiment configuration selector.
    --var_name CHANNEL         Required input channel to permute (for example, sst_lag1).
    --random_seed INTEGER      Optional seed for a reproducible permutation.
    --overwrite                Replace an existing permutation-prediction output.

Examples:
    python -m src.models.permute_and_predict --config exp1_inputs:input4 --var_name z500_lag1 --random_seed 7
    python -m src.models.permute_and_predict --config exp1_inputs:input3e --var_name z50_lag6 --overwrite
"""

import numpy as np
import xarray as xr
import pandas as pd
import os
from tqdm import tqdm
import torch
import argparse

from src import config_cesm
from src.utils import util_cesm
from src.models.models import UNetRes3
from src.experiment_configs import load_config
from src.utils.util_shared import write_nc_file
from src.models.evaluate import nn_ens_members
from src.models.models_util import CESM_Dataset

def permute_ds(
    ds: xr.Dataset, 
    var_name: str, 
    random_seed: int | None = None
) -> xr.Dataset:
    """
    Permute the specified variable in time.
    """
    rng = np.random.default_rng(random_seed)
    perm = rng.permutation(ds.sizes["start_prediction_month"])
    permuted_ds = ds.copy()
    permuted = ds.sel(channel=var_name).isel(start_prediction_month=perm).assign_coords(
        start_prediction_month = ds.start_prediction_month
    )

    permuted_ds.loc[dict(channel=var_name)] = permuted
    return permuted_ds

def main():
    parser = argparse.ArgumentParser(description="Permute a variable in the test set data.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Named configuration selector (e.g., exp1_inputs:input2)",
    )
    parser.add_argument(
        "--var_name",
        type=str,
        required=True,
        help="Name of the variable to permute (e.g., 'sst')",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Random seed for permutation",
    )
    parser.add_argument("--overwrite", action="store_true", help="If set, overwrite existing output files.")
    args = parser.parse_args()
    config = load_config(args.config)

    # Initialize an empty dataset 
    if config.data_split["split_by"] == "ensemble_member":
        ensemble_members = config.data_split["test"]
        time_coords = config.data_split["time_range"]
    elif config.data_split["split_by"] == "time":
        ensemble_members = config.data_split["member_ids"]
        time_coords = config.data_split["test"]
    num_members = len(ensemble_members)
    channels, x_dim, y_dim = config.max_lead_months, 80, 80
    reference_grid = util_cesm.generate_sps_grid()
    test_dataset = CESM_Dataset("test", config)
    permutation_indices = {}
    for member_id in ensemble_members:
        rng = np.random.default_rng(args.random_seed)
        permutation_indices[member_id] = rng.permutation(len(time_coords))

    # number of trained ensemble members (nn_member_id). Note that this is different than the
    # member_id, which refers to the CESM ensemble member on which we are evaluating 
    num_nn_members = len(nn_ens_members(config))

    ds = xr.Dataset(
        {
            "predictions": (
                ["start_prediction_month", "member_id", "nn_member_id", "lead_time", "y", "x"],
                np.full((len(time_coords), num_members, num_nn_members, channels, y_dim, x_dim), 
                        np.nan, dtype=np.float32)
            )
        },
        coords={
            "start_prediction_month": time_coords,
            "member_id": ensemble_members,
            "nn_member_id": np.arange(num_nn_members),
            "lead_time": np.arange(1, channels + 1),
            "y": reference_grid.y.values,
            "x": reference_grid.x.values,
        }
    )

    in_channels = util_cesm.get_num_input_channels(config.input_config)
    out_channels = util_cesm.get_num_output_channels(config.max_lead_months, config.target_config)
    
    # Load model architecture
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.model == "UNetRes3":
        model = UNetRes3(in_channels=in_channels, 
                        out_channels=out_channels, 
                        predict_anomalies=config.target_config["predict_anom"],
                        **config.model_args).to(device)
    else: 
        raise NotImplementedError(f"Model {config.model} not implemented.")
    
    for nn_member_idx, filename in enumerate(nn_ens_members(config)):
        checkpoint_path = os.path.join(config_cesm.MODEL_DIRECTORY, config.experiment_name, filename)

        if not os.path.exists(checkpoint_path): 
            raise FileNotFoundError(f"No checkpoint file found at {checkpoint_path}")

        print(f"Evaluating checkpoint at {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # Populate the Dataset with predictions
        with torch.no_grad():
            for member_idx, member_id in enumerate(ensemble_members):
                print(f"Permuting variable {args.var_name} for member {member_id}...")
                for time_idx, start_prediction_month in enumerate(
                    tqdm(time_coords, desc=f"Member {member_id}")
                ):
                    input_data = test_dataset.input_data_array(
                        member_id, start_prediction_month
                    ).load().copy()
                    source_time = time_coords[
                        permutation_indices[member_id][time_idx]
                    ]
                    source_data = test_dataset.input_data_array(
                        member_id, source_time
                    )
                    if args.var_name not in input_data.channel.values:
                        raise ValueError(
                            f"Unknown input channel {args.var_name!r}; "
                            f"available channels: "
                            f"{input_data.channel.values.tolist()}"
                        )
                    input_data.loc[{"channel": args.var_name}] = source_data.sel(
                        channel=args.var_name
                    )
                    input_tensor = torch.tensor(
                        input_data.values,
                        dtype=torch.float32,
                        device=device
                    ).unsqueeze(0)
                    predictions = model(input_tensor).cpu().numpy()[0]  # Move predictions to CPU
                    ds["predictions"][time_idx, member_idx, nn_member_idx, :, :, :] = predictions
                
    # Save the Dataset as NetCDF
    output_dir = os.path.join(config_cesm.PREDICTIONS_DIRECTORY, config.experiment_name, "permute")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"permute_{args.var_name}_predictions.nc")
    write_nc_file(ds, output_path, overwrite=args.overwrite)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    main()