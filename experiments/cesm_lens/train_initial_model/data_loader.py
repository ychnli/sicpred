import xarray as xr
import torch
import os

class CESM_SeaIceDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, ensemble_members, split, data_split_settings, transform=None):
        """
        Param:
            (string)        data_dir (preprocessed model-ready data)
            (list)          ensemble_members (ripf notation)
            (string)        split ('test', 'val', or 'train')
            (dict)          data_split_settings
            (callable)      optional transform
        """

        self.data_dir = data_dir
        self.ensemble_members = ensemble_members
        self.split = split
        self.transform = transform

        # Extract split settings
        self.split_by = data_split_settings["split_by"]  # 'time' in this case
        self.split_range = data_split_settings[split]   # Date range for the requested split

        # Build a global index of samples
        self.samples = []
        for member_id in ensemble_members:
            input_file = os.path.join(data_dir, f"inputs_member_{member_id}.nc")
            with xr.open_dataset(input_file) as ds:
                time_values = ds["start_prediction_month"].values
                for start_idx, time_val in enumerate(time_values):
                    # Only include samples within the specified date range for the split
                    if time_val in self.split_range:
                        self.samples.append((member_id, time_val, start_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        member_id, start_prediction_month, start_idx = self.samples[idx]
        input_file = os.path.join(self.data_dir, f"inputs_member_{member_id}.nc")
        target_file = os.path.join(self.data_dir, f"targets_member_{member_id}.nc")

        # Load the specific sample lazily
        with xr.open_dataset(input_file) as input_ds:
            input_sample = input_ds["data"].isel(start_prediction_month=start_idx)

        with xr.open_dataset(target_file) as target_ds:
            target_sample = target_ds["data"].isel(start_prediction_month=start_idx)
        
        start_prediction_month = input_sample.start_prediction_month.values
        time_npy = np.array([start_prediction_month.year, start_prediction_month.month]) 

        sample = {
            "input": torch.tensor(input_sample.values, dtype=torch.float32),
            "target": torch.tensor(target_sample.values, dtype=torch.float32),
            "start_prediction_month": time_npy,
            "member_id": member_id
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
