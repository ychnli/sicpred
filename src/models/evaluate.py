import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import argparse
import xarray as xr
import re
from tqdm import tqdm  

from src.models.models_util import CESM_Dataset
from src.models.models import UNetRes3
from src.utils import util_cesm
from src.utils import util_shared
from src import config_cesm
from src.experiment_configs import load_config

def nn_ens_members(config): 
    """ Given a config file, returns a list of checkpoint files for trained ensemble members (diff training
        initializations), using a regex """

    files = os.listdir(os.path.join(config_cesm.MODEL_DIRECTORY, config.experiment_name))
    pattern = re.compile(
        rf"{config.model}_{config.experiment_name}_member_\d+_{config.checkpoint_to_evaluate}\.pth"
    )
    return sorted([filename for filename in files if pattern.match(filename)])

import argparse
import os

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import DataLoader
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Train a model with specified config.")
    parser.add_argument("--config", type=str, required=True,
                        help="Named configuration selector (e.g., exp1_inputs:input2)")
    parser.add_argument("--device", type=str, help="cuda or cpu")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--overwrite", action="store_true",
                        help="If set, overwrite existing output files.")
    parser.add_argument("--zero-shot", type=str, default=None,
                        help="Named configuration selector for a dataset that the model has not seen "
                             "to evaluate the model on (e.g., evaluating a model pretrained on CESM "
                             "to zero-shot predict obs)")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    # Load configurations
    config = load_config(args.config)
    if args.zero_shot is not None:
        config_zs = load_config(args.zero_shot)
        data_split_settings = config_zs.data_split
        dataset_config = config_zs
    else:
        config_zs = None
        data_split_settings = config.data_split
        dataset_config = config

    # Construct save paths and check if exists
    if args.zero_shot is not None:
        output_dir = os.path.join(config_cesm.PREDICTIONS_DIRECTORY, config_zs.experiment_name)
        output_path = os.path.join(
            output_dir,
            f"{config.experiment_name}_{config.model}_zeroshot_predictions.nc"
        )
    else:
        output_dir = os.path.join(config_cesm.PREDICTIONS_DIRECTORY, config.experiment_name)
        output_path = os.path.join(
            output_dir,
            f"{config.model}_{config.checkpoint_to_evaluate}_predictions.nc"
        )

    os.makedirs(output_dir, exist_ok=True)
    if not args.overwrite and os.path.exists(output_path):
        print(f"Info: found existing path {output_path}. Model evaluation skipped because overwrite was off")
        return

    # Dataset / dataloader
    test_dataset = CESM_Dataset(args.split, dataset_config)
    use_cuda = (args.device == "cuda") or (args.device is None and torch.cuda.is_available())
    if args.device in ["cuda", "cpu"]:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        persistent_workers=(args.num_workers > 0),
    )

    in_channels = util_cesm.get_num_input_channels(config.input_config)
    out_channels = util_cesm.get_num_output_channels(config.max_lead_months, config.target_config)

    # Load model architecture
    if config.model == "UNetRes3":
        model = UNetRes3(
            in_channels=in_channels,
            out_channels=out_channels,
            predict_anomalies=config.target_config["predict_anom"],
            **config.model_args,
        ).to(device)
    else:
        raise NotImplementedError(f"Model {config.model} not implemented.")

    # Figure out output coordinates
    if data_split_settings["split_by"] == "ensemble_member":
        if args.split == "all":
            ensemble_members = [
                *data_split_settings["train"],
                *data_split_settings["val"],
                *data_split_settings["test"],
            ]
        else:
            ensemble_members = data_split_settings["test"]
        time_coords = data_split_settings["time_range"]

    elif data_split_settings["split_by"] == "time":
        ensemble_members = data_split_settings["member_ids"]
        if args.split == "all":
            time_coords = (
                data_split_settings["train"]
                .union(data_split_settings["val"])
                .union(data_split_settings["test"])
            )
        else:
            time_coords = data_split_settings["test"]
    else:
        raise ValueError(f"Unsupported split_by = {data_split_settings['split_by']}")

    time_coords = list(time_coords)
    ensemble_members = list(ensemble_members)

    num_members = len(ensemble_members)
    num_nn_members = len(nn_ens_members(config))
    channels, x_dim, y_dim = config.max_lead_months, 80, 80
    reference_grid = util_cesm.generate_sps_grid()

    # Build output dataset
    ds = xr.Dataset(
        {
            "predictions": (
                ["start_prediction_month", "member_id", "nn_member_id", "lead_time", "y", "x"],
                np.full(
                    (len(time_coords), num_members, num_nn_members, channels, y_dim, x_dim),
                    np.nan,
                    dtype=np.float32,
                ),
            )
        },
        coords={
            "start_prediction_month": time_coords,
            "member_id": ensemble_members,
            "nn_member_id": np.arange(num_nn_members),
            "lead_time": np.arange(1, channels + 1),
            "y": reference_grid.y.values,
            "x": reference_grid.x.values,
        },
    )

    pred_array = ds["predictions"].values

    time_to_idx = {pd.Timestamp(t): i for i, t in enumerate(time_coords)}
    member_to_idx = {m: i for i, m in enumerate(ensemble_members)}

    for nn_member_idx, filename in enumerate(nn_ens_members(config)):
        checkpoint_path = os.path.join(config_cesm.MODEL_DIRECTORY, config.experiment_name, filename)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint file found at {checkpoint_path}")

        print(f"Evaluating checkpoint at {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc=f"Evaluating nn member {nn_member_idx}", unit="batch"):
                inputs = batch["input"].to(device, non_blocking=True)

                batch_preds = model(inputs)
                batch_preds = batch_preds.detach().cpu().numpy()  # shape: (B, lead_time, y, x)

                # where [:, 0, 0] = year and [:, 0, 1] = month
                spm = batch["start_prediction_month"]
                if isinstance(spm, torch.Tensor):
                    spm_np = spm.cpu().numpy()
                else:
                    spm_np = np.asarray(spm)

                years = spm_np[:, 0, 0]
                months = spm_np[:, 0, 1]

                # Convert batch metadata to output indices
                batch_time_idxs = np.fromiter(
                    (
                        time_to_idx[pd.Timestamp(year=int(y), month=int(m), day=1)]
                        for y, m in zip(years, months)
                    ),
                    dtype=np.int64,
                    count=len(years),
                )

                member_ids = batch["member_id"]
                batch_member_idxs = np.fromiter(
                    (member_to_idx[m] for m in member_ids),
                    dtype=np.int64,
                    count=len(member_ids),
                )

                pred_array[batch_time_idxs, batch_member_idxs, nn_member_idx, :, :, :] = batch_preds

    util_shared.write_nc_file(ds, output_path, overwrite=args.overwrite)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    main()