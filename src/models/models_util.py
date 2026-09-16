import xarray as xr
import dask
import numpy as np
import pandas as pd
from time import time
import os
import pickle
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src import config_cesm
from src.utils import util_cesm

##########################################################################################
# Model utils
##########################################################################################

class CESM_Dataset(torch.utils.data.Dataset):
    """Load model-ready CESM input-target pairs created during preprocessing."""

    def __init__(self, split, experiment_config):
        self.config = experiment_config
        self.data_split_settings = experiment_config.data_split
        self.data_dir = os.path.join(
            config_cesm.PROCESSED_DATA_DIRECTORY,
            "data_pairs",
            experiment_config.data_name,
        )
        self.split = split

        if split not in {"train", "val", "test", "all"}:
            raise ValueError("split must be 'train', 'val', 'test', or 'all'")

        split_by = self.data_split_settings["split_by"]
        if split_by == "time":
            member_ids = list(self.data_split_settings["member_ids"])
            if split == "all":
                prediction_months = util_cesm.get_start_prediction_months(
                    self.data_split_settings
                )
            else:
                prediction_months = self.data_split_settings[split]
        elif split_by == "ensemble_member":
            prediction_months = self.data_split_settings["time_range"]
            if split == "all":
                member_ids = [
                    *self.data_split_settings["train"],
                    *self.data_split_settings["val"],
                    *self.data_split_settings["test"],
                ]
            else:
                member_ids = list(self.data_split_settings[split])
        else:
            raise ValueError(f"Unsupported split_by={split_by!r}")

        allowed_months = set(pd.DatetimeIndex(prediction_months))
        self.samples = []
        for member_id in member_ids:
            input_path = self._pair_path("inputs", member_id)
            if not os.path.exists(input_path):
                raise FileNotFoundError(
                    f"Missing preprocessed inputs at {input_path}. "
                    "Run src.preprocessing.preprocess_cesm_data first."
                )
            with xr.open_dataset(input_path) as input_ds:
                time_values = pd.DatetimeIndex(
                    input_ds["start_prediction_month"].values
                )
            self.samples.extend(
                (member_id, pd.Timestamp(month), index)
                for index, month in enumerate(time_values)
                if month in allowed_months
            )

    def _pair_path(self, kind, member_id):
        return os.path.join(self.data_dir, f"{kind}_member_{member_id}.nc")

    def input_data_array(self, member_id, start_prediction_month):
        """Load one precomputed input sample by member and initialization month."""
        path = self._pair_path("inputs", member_id)
        with xr.open_dataset(path) as dataset:
            return dataset["data"].sel(
                start_prediction_month=pd.Timestamp(start_prediction_month)
            ).load()

    def target_data_array(self, member_id, start_prediction_month):
        """Load one precomputed target sample by member and initialization month."""
        path = self._pair_path("targets", member_id)
        with xr.open_dataset(path) as dataset:
            return dataset["data"].sel(
                start_prediction_month=pd.Timestamp(start_prediction_month)
            ).load()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        member_id, start_prediction_month, start_idx = self.samples[idx]
        with xr.open_dataset(self._pair_path("inputs", member_id)) as input_ds:
            input_sample = input_ds["data"].isel(
                start_prediction_month=start_idx
            ).load()
        with xr.open_dataset(self._pair_path("targets", member_id)) as target_ds:
            target_sample = target_ds["data"].isel(
                start_prediction_month=start_idx
            ).load()

        target_months = pd.date_range(
            start_prediction_month,
            start_prediction_month
            + pd.DateOffset(months=target_sample.sizes["lead_time"] - 1),
            freq="MS",
        )
        return {
            "input": torch.tensor(input_sample.values, dtype=torch.float32),
            "target": torch.tensor(target_sample.values, dtype=torch.float32),
            "start_prediction_month": np.column_stack(
                (target_months.year, target_months.month)
            ),
            "member_id": member_id,
        }



def _split_members_and_months(data_split_settings, split):
    """Return ordered ensemble members and initialization months for a split."""
    if split not in {"train", "val", "test", "all"}:
        raise ValueError("split must be 'train', 'val', 'test', or 'all'")

    split_by = data_split_settings["split_by"]
    if split_by == "time":
        member_ids = list(data_split_settings["member_ids"])
        if split == "all":
            prediction_months = util_cesm.get_start_prediction_months(
                data_split_settings
            )
        else:
            prediction_months = data_split_settings[split]
    elif split_by == "ensemble_member":
        prediction_months = data_split_settings["time_range"]
        if split == "all":
            member_ids = [
                *data_split_settings["train"],
                *data_split_settings["val"],
                *data_split_settings["test"],
            ]
        else:
            member_ids = list(data_split_settings[split])
    else:
        raise ValueError(f"Unsupported split_by={split_by!r}")

    return member_ids, pd.DatetimeIndex(prediction_months)


def _ordered_union(sequences):
    """Return the first-seen union of several ordered sequences."""
    return list(dict.fromkeys(item for sequence in sequences for item in sequence))


class EagerCESMDataStore:
    """Eagerly load normalized physical fields shared by dataset views.

    Arrays are converted once to contiguous float32 NumPy storage. Dataset
    workers created with ``fork`` consequently share these read-only arrays by
    copy-on-write instead of independently reopening NetCDF files.
    """

    def __init__(self, experiment_config, splits=("train",), include_targets=True):
        self.config = experiment_config
        self.include_targets = include_targets
        split_members = [
            _split_members_and_months(experiment_config.data_split, split)[0]
            for split in splits
        ]
        self.member_ids = _ordered_union(split_members)
        self.member_positions = {
            member_id: index for index, member_id in enumerate(self.member_ids)
        }
        self.input_specs = []
        self.arrays = {}
        self.time_positions = {}
        self.load_seconds = {}
        self.x_coords = None
        self.y_coords = None

        opened_arrays = util_cesm.load_inputs_data_da_dict(
            experiment_config.input_config, experiment_config.data_split
        )
        try:
            for name, settings in experiment_config.input_config.items():
                if not settings["include"]:
                    continue
                if settings["auxiliary"]:
                    self.input_specs.append(("auxiliary", name, 1))
                    continue

                self.input_specs.append(("physical", name, settings["lag"]))
                self._load_array(name, opened_arrays[name])
        finally:
            for data_array in opened_arrays.values():
                close = getattr(data_array, "close", None)
                if close is not None:
                    close()

        if not self.arrays:
            raise ValueError("At least one physical input variable must be enabled")

        self.target_array = None
        self.target_time_positions = None
        if include_targets:
            if (
                experiment_config.target_config["predict_anom"]
                and "icefrac" in self.arrays
            ):
                self.target_array = self.arrays["icefrac"]
                self.target_time_positions = self.time_positions["icefrac"]
            else:
                self._load_target_array()

        if any(name == "land_mask" for _, name, _ in self.input_specs):
            with xr.open_dataset(util_cesm.LAND_MASK_PATH) as land_mask_ds:
                self.land_mask = np.ascontiguousarray(
                    land_mask_ds["mask"].values, dtype=np.float32
                )
            expected_shape = (len(self.y_coords), len(self.x_coords))
            if self.land_mask.shape != expected_shape:
                raise ValueError(
                    f"Unexpected land-mask shape {self.land_mask.shape}; "
                    f"expected {expected_shape}"
                )
        else:
            self.land_mask = None

        separate_target_bytes = (
            0
            if self.target_array is None
            or self.target_array is self.arrays.get("icefrac")
            else self.target_array.nbytes
        )
        self.resident_data_gib = (
            sum(array.nbytes for array in self.arrays.values())
            + separate_target_bytes
            + (0 if self.land_mask is None else self.land_mask.nbytes)
        ) / 1024**3

    def _prepare_array(self, name, data_array):
        selected = data_array.sel(member_id=self.member_ids)
        for dimension in ("z_t", "lev"):
            if dimension not in selected.dims:
                continue
            if selected.sizes[dimension] != 1:
                raise ValueError(
                    f"Input dimension {dimension!r} for {name!r} must be scalar, "
                    f"got {selected.sizes[dimension]} values"
                )
            selected = selected.isel({dimension: 0}, drop=True)
        return selected.transpose("member_id", "time", "y", "x")

    def _load_array(self, name, data_array):
        selected = self._prepare_array(name, data_array)
        started = time()
        with dask.config.set(scheduler="synchronous"):
            selected.load()
        self.arrays[name] = np.ascontiguousarray(
            selected.values, dtype=np.float32
        )
        self.load_seconds[name] = time() - started
        self.time_positions[name] = {
            pd.Timestamp(value): index
            for index, value in enumerate(selected.time.values)
        }

        x_coords = np.asarray(selected.x.values)
        y_coords = np.asarray(selected.y.values)
        if self.x_coords is None:
            self.x_coords = x_coords
            self.y_coords = y_coords
        elif not (
            np.array_equal(self.x_coords, x_coords)
            and np.array_equal(self.y_coords, y_coords)
        ):
            raise ValueError(f"Spatial coordinates differ for {name!r}")

    def _load_target_array(self):
        if self.config.target_config["predict_anom"]:
            target_path = os.path.join(
                config_cesm.PROCESSED_DATA_DIRECTORY,
                "normalized_inputs",
                self.config.data_split["name"],
                "icefrac_norm.nc",
            )
        elif self.config.data_split["member_ids"] == ["obs"]:
            target_path = os.path.join(
                config_cesm.DATA_DIRECTORY, "obs_data", "icefrac_obs.nc"
            )
        else:
            target_path = os.path.join(
                config_cesm.DATA_DIRECTORY,
                "cesm_data",
                "icefrac",
                "icefrac_combined.nc",
            )

        with xr.open_dataset(target_path, chunks={"member_id": 1}) as target_ds:
            target = target_ds["icefrac"]
            if "month" in target.coords:
                target = target.drop_vars("month")
            selected = self._prepare_array("icefrac target", target)
            started = time()
            with dask.config.set(scheduler="synchronous"):
                selected.load()
            self.target_array = np.ascontiguousarray(
                selected.values, dtype=np.float32
            )
            self.load_seconds["target:icefrac"] = time() - started
            self.target_time_positions = {
                pd.Timestamp(value): index
                for index, value in enumerate(selected.time.values)
            }


class EagerDynamicCESMDataset(torch.utils.data.Dataset):
    """Construct model samples from normalized fields already resident in RAM."""

    def __init__(self, split, experiment_config, *, store=None, include_targets=True):
        self.config = experiment_config
        self.split = split
        self.include_targets = include_targets
        self.store = store or EagerCESMDataStore(
            experiment_config,
            splits=(split,),
            include_targets=include_targets,
        )
        if include_targets and self.store.target_array is None:
            raise ValueError("The resident store was created without targets")
        self.member_ids, start_months = _split_members_and_months(
            experiment_config.data_split, split
        )
        missing_members = set(self.member_ids) - set(self.store.member_ids)
        if missing_members:
            raise ValueError(
                f"Resident store does not contain members {sorted(missing_members)}"
            )

        self.channel_names = []
        for kind, name, lag in self.store.input_specs:
            if kind == "physical":
                self.channel_names.extend(
                    f"{name}_lag{lag_index}"
                    for lag_index in range(lag, 0, -1)
                )
            else:
                self.channel_names.append(name)

        self.month_metadata = {}
        for start_month in start_months:
            start_month = pd.Timestamp(start_month)
            input_slices = {}
            for kind, name, lag in self.store.input_specs:
                if kind != "physical":
                    continue
                months = pd.date_range(
                    start_month - pd.DateOffset(months=lag),
                    start_month - pd.DateOffset(months=1),
                    freq="MS",
                )
                input_slices[name] = self._contiguous_slice(
                    self.store.time_positions[name], months, name
                )

            target_months = pd.date_range(
                start_month,
                start_month
                + pd.DateOffset(months=experiment_config.max_lead_months - 1),
                freq="MS",
            )
            target_slice = None
            if self.include_targets:
                target_slice = self._contiguous_slice(
                    self.store.target_time_positions,
                    target_months,
                    "icefrac target",
                )
            self.month_metadata[start_month] = (
                start_month.month,
                np.column_stack((target_months.year, target_months.month)),
                input_slices,
                target_slice,
            )

        self.samples = [
            (self.store.member_positions[member_id], member_id, start_month)
            for member_id in self.member_ids
            for start_month in start_months
        ]

    @staticmethod
    def _contiguous_slice(time_positions, months, field_name):
        try:
            positions = [time_positions[pd.Timestamp(month)] for month in months]
        except KeyError as exc:
            raise KeyError(
                f"Missing required month {pd.Timestamp(exc.args[0])} "
                f"for {field_name!r}"
            ) from exc
        expected = list(range(positions[0], positions[0] + len(positions)))
        if positions != expected:
            raise ValueError(f"Required months are not contiguous for {field_name!r}")
        return slice(positions[0], positions[-1] + 1)

    def __len__(self):
        return len(self.samples)

    def _input_numpy(self, member_position, start_month):
        init_month, _, input_slices, _ = self.month_metadata[start_month]
        parts = []
        for kind, name, _ in self.store.input_specs:
            if kind == "physical":
                parts.append(
                    self.store.arrays[name][member_position, input_slices[name]]
                )
            elif name == "cosine_of_init_month":
                parts.append(
                    np.full(
                        (1, len(self.store.y_coords), len(self.store.x_coords)),
                        np.cos(2 * np.pi * init_month / 12),
                        dtype=np.float32,
                    )
                )
            elif name == "sine_of_init_month":
                parts.append(
                    np.full(
                        (1, len(self.store.y_coords), len(self.store.x_coords)),
                        np.sin(2 * np.pi * init_month / 12),
                        dtype=np.float32,
                    )
                )
            elif name == "land_mask":
                parts.append(self.store.land_mask[None, :, :])
            else:
                raise NotImplementedError(f"Unknown auxiliary input {name!r}")

        sample = np.concatenate(parts, axis=0)
        sample[np.isnan(sample)] = 0
        return sample

    def _target_numpy(self, member_position, start_month):
        if not self.include_targets:
            raise RuntimeError("This dataset was created without targets")
        target_slice = self.month_metadata[start_month][3]
        sample = self.store.target_array[member_position, target_slice].copy()
        sample[np.isnan(sample)] = 0
        return sample

    def _sample_coordinates(self, member_id, start_prediction_month):
        start_prediction_month = pd.Timestamp(start_prediction_month)
        try:
            member_position = self.store.member_positions[member_id]
            metadata = self.month_metadata[start_prediction_month]
        except KeyError as exc:
            raise KeyError(
                f"Sample ({member_id!r}, {start_prediction_month}) is not in "
                f"the {self.split!r} dataset"
            ) from exc
        return member_position, metadata

    def input_data_array(self, member_id, start_prediction_month):
        """Construct one labeled input sample without accessing disk."""
        start_prediction_month = pd.Timestamp(start_prediction_month)
        member_position, _ = self._sample_coordinates(
            member_id, start_prediction_month
        )
        return xr.DataArray(
            self._input_numpy(member_position, start_prediction_month),
            dims=("channel", "y", "x"),
            coords={
                "channel": self.channel_names,
                "member_id": member_id,
                "y": self.store.y_coords,
                "x": self.store.x_coords,
                "start_prediction_month": start_prediction_month,
            },
            name="data",
        )

    def target_data_array(self, member_id, start_prediction_month):
        """Construct one labeled target sample without accessing disk."""
        start_prediction_month = pd.Timestamp(start_prediction_month)
        member_position, _ = self._sample_coordinates(
            member_id, start_prediction_month
        )
        return xr.DataArray(
            self._target_numpy(member_position, start_prediction_month),
            dims=("lead_time", "y", "x"),
            coords={
                "lead_time": np.arange(1, self.config.max_lead_months + 1),
                "member_id": member_id,
                "y": self.store.y_coords,
                "x": self.store.x_coords,
                "start_prediction_month": start_prediction_month,
            },
            name="data",
        )

    def __getitem__(self, idx):
        member_position, member_id, start_month = self.samples[idx]
        target_months = self.month_metadata[start_month][1]
        sample = {
            "input": torch.from_numpy(
                self._input_numpy(member_position, start_month)
            ),
            "start_prediction_month": target_months,
            "member_id": member_id,
        }
        if self.include_targets:
            sample["target"] = torch.from_numpy(
                self._target_numpy(member_position, start_month)
            )
        return sample


def build_cesm_dataset(
    split, experiment_config, *, data_source="dynamic", store=None, include_targets=True
):
    """Build either the dynamic (default) or precomputed-pair dataset."""
    if data_source == "dynamic":
        return EagerDynamicCESMDataset(
            split,
            experiment_config,
            store=store,
            include_targets=include_targets,
        )
    if data_source == "precomputed":
        if store is not None:
            raise ValueError("A resident store cannot be used with precomputed pairs")
        return CESM_Dataset(split, experiment_config)
    raise ValueError("data_source must be 'dynamic' or 'precomputed'")



def load_cesm_targets_data_array(
    split, experiment_config, *, data_source="dynamic", chunks=None
):
    """Load all targets for diagnostics without per-sample file access.

    When chunks are provided, the dynamic-data path remains lazy so callers can
    reduce diagnostics a block at a time instead of materializing the full
    overlapping lead-time tensor in memory.
    """
    member_ids, start_months = _split_members_and_months(
        experiment_config.data_split, split
    )
    if data_source == "precomputed":
        dataset = CESM_Dataset(split, experiment_config)
        member_arrays = []
        for member_id in member_ids:
            member_arrays.append(
                xr.concat(
                    [
                        dataset.target_data_array(member_id, start_month)
                        for start_month in start_months
                    ],
                    dim="start_prediction_month",
                )
            )
        result = xr.concat(member_arrays, dim="member_id").assign_coords(
            member_id=member_ids
        ).transpose(
            "start_prediction_month", "member_id", "lead_time", "y", "x"
        ).load()
        if chunks is not None:
            result = result.chunk(
                {dim: size for dim, size in chunks.items() if dim in result.dims}
            )
        return result
    if data_source != "dynamic":
        raise ValueError("data_source must be 'dynamic' or 'precomputed'")

    if experiment_config.target_config["predict_anom"]:
        target_path = os.path.join(
            config_cesm.PROCESSED_DATA_DIRECTORY,
            "normalized_inputs",
            experiment_config.data_split["name"],
            "icefrac_norm.nc",
        )
    elif experiment_config.data_split["member_ids"] == ["obs"]:
        target_path = os.path.join(
            config_cesm.DATA_DIRECTORY, "obs_data", "icefrac_obs.nc"
        )
    else:
        target_path = os.path.join(
            config_cesm.DATA_DIRECTORY,
            "cesm_data",
            "icefrac",
            "icefrac_combined.nc",
        )

    source_chunks = {"member_id": 1} if chunks is None else {
        "member_id": 1,
        "time": chunks.get("start_prediction_month", 12),
        "y": -1,
        "x": -1,
    }
    target_ds = xr.open_dataset(target_path, chunks=source_chunks)
    target = target_ds["icefrac"]
    if "month" in target.coords:
        target = target.drop_vars("month")
    target = target.sel(member_id=member_ids).transpose(
        "member_id", "time", "y", "x"
    )
    time_positions = {
        pd.Timestamp(value): index
        for index, value in enumerate(target.time.values)
    }
    target_indices = np.asarray(
        [
            [
                time_positions[month]
                for month in pd.date_range(
                    start_month,
                    start_month
                    + pd.DateOffset(
                        months=experiment_config.max_lead_months - 1
                    ),
                    freq="MS",
                )
            ]
            for start_month in start_months
        ]
    )

    if chunks is None:
        with dask.config.set(scheduler="synchronous"):
            target.load()
        values = np.asarray(target.values)[:, target_indices].transpose(
            1, 0, 2, 3, 4
        )
        values = values.copy()
        values[np.isnan(values)] = 0
    else:
        indexer = xr.DataArray(
            target_indices,
            dims=("start_prediction_month", "lead_time"),
        )
        values = target.isel(time=indexer).transpose(
            "start_prediction_month", "member_id", "lead_time", "y", "x"
        ).data

    result = xr.DataArray(
        values,
        dims=(
            "start_prediction_month",
            "member_id",
            "lead_time",
            "y",
            "x",
        ),
        coords={
            "start_prediction_month": start_months,
            "member_id": member_ids,
            "lead_time": np.arange(
                1, experiment_config.max_lead_months + 1
            ),
            "y": target.y.values,
            "x": target.x.values,
        },
        name="data",
    ).fillna(0)
    if chunks is None:
        target_ds.close()
    else:
        result = result.chunk(
            {dim: size for dim, size in chunks.items() if dim in result.dims}
        )
        result.set_close(target_ds.close)
    return result


def build_cesm_dataloader(
    dataset,
    *,
    batch_size,
    shuffle,
    num_workers=2,
    pin_memory=False,
    prefetch_factor=2,
    generator=None,
):
    """Build a loader that overlaps CPU construction with model computation."""
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "generator": generator,
    }
    if num_workers > 0:
        kwargs.update(
            persistent_workers=True,
            prefetch_factor=prefetch_factor,
            multiprocessing_context="fork",
        )
    return DataLoader(dataset, **kwargs)


def prime_cesm_dataloader(dataloader):
    """Start persistent workers before CUDA initialization.

    Forking while only CPU state exists lets workers share eager arrays safely.
    A subsequent ``iter(dataloader)`` resets this persistent iterator normally.
    """
    if dataloader.num_workers > 0:
        iter(dataloader)


class Obs_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_directory, configuration, split_array, start_prediction_months, \
                split_type='train', target_shape=(336, 320), mode="regression", class_splits=None):
        self.data_directory = data_directory
        self.configuration = configuration
        self.split_array = split_array
        self.start_prediction_months = start_prediction_months
        self.split_type = split_type
        self.target_shape = target_shape
        self.class_splits = class_splits
        self.mode = mode

        # Open the HDF5 files
        self.inputs_file = h5py.File(f"{data_directory}/inputs_{configuration}.h5", 'r')

        if "sicanom" in configuration: 
            targets_configuration = "anom_regression" 
        else: 
            targets_configuration = "regression"

        self.targets_file = h5py.File(f"{data_directory}/targets_{targets_configuration}.h5", 'r')
        
        self.inputs = self.inputs_file[f"inputs_{configuration}"]
        self.targets = self.targets_file['targets_sea_ice_only']

        self.n_samples, self.n_channels, self.n_y, self.n_x = self.inputs.shape
        
        # Get indices for the specified split type
        if isinstance(split_type, str): 
            self.indices = np.where(split_array == split_type)[0]
        elif isinstance(split_type, list):
            self.indices = np.where(np.isin(split_array, split_type))[0]
        else:
            raise TypeError("split_type needs to be one of str or list")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        input_data = self.inputs[actual_idx]
        target_data = self.targets[actual_idx]
        start_prediction_month = self.start_prediction_months[actual_idx]

        # Pad input_data and target_data to the target shape
        pad_y = self.target_shape[0] - self.n_y
        pad_x = self.target_shape[1] - self.n_x
        input_data = np.pad(input_data, ((0, 0), (pad_y//2, pad_y//2), (pad_x//2, pad_x//2)), mode='constant', constant_values=0)
        target_data = np.pad(target_data, ((0, 0), (pad_y//2, pad_y//2), (pad_x//2, pad_x//2)), mode='constant', constant_values=0)

        # If we are doing classification, then discretise the target data
        if self.mode == "classification":
            if self.class_splits is None:
                raise ValueError("need to specify a monotonically increasing list class_splits denoting class boundaries")

            # check if class_split is monotonically increasing
            if len(self.class_splits) > 1 and np.any(np.diff(self.class_splits) < 0): 
                raise ValueError("class_splits needs to be monotonically increasing")

            bounds = [] # bounds for classes
            for i,class_split in enumerate(self.class_splits): 
                if i == 0: 
                    bounds.append([0, class_split])
                if i == len(self.class_splits) - 1: 
                    bounds.append([class_split, 1])
                else: 
                    bounds.append([class_split, self.class_splits[i+1]])
            
            target_classes_data = np.zeros_like(target_data) 
            target_classes_data = target_classes_data[np.newaxis,:,:,:]
            target_classes_data = np.repeat(target_classes_data, len(bounds), axis=0)
            for i,bound in enumerate(bounds): 
                if i == len(bounds) - 1: 
                    target_classes_data[i,:,:,:] = np.logical_and(target_data >= bound[0], target_data <= bound[1]).astype(int)
                else:
                    target_classes_data[i,:,:,:] = np.logical_and(target_data >= bound[0], target_data < bound[1]).astype(int)
            
            target_data = target_classes_data 

        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        target_tensor = torch.tensor(target_data, dtype=torch.float32)

        # Get the target months for this sample
        target_months = pd.date_range(start=start_prediction_month, end=start_prediction_month + pd.DateOffset(months=5), freq="MS")
        target_months = target_months.month.to_numpy()
        
        return input_tensor, target_tensor, target_months

    def __del__(self):
        self.inputs_file.close()
        self.targets_file.close()



def print_split_stats(split_array):
    ntrain = sum(split_array == 'train')
    nval = sum(split_array == 'val')
    ntest = sum(split_array == 'test')
    
    print(f"train samples: {ntrain} ({round(ntrain / len(split_array), 2)})")
    print(f"val samples: {nval} ({round(nval / len(split_array), 2)})")
    print(f"test samples: {ntest} ({round(ntest / len(split_array), 2)})")


def generate_start_prediction_months(max_month_lead_time=6, max_input_lag_time=12):
    # Construct the date range for the data pairs 
    # Note that this is not continuous due to the missing data in 1987-1988 
    first_range = pd.date_range('1981-01', pd.Timestamp('1987-12') - pd.DateOffset(months=max_month_lead_time+1), freq='MS')
    second_range = pd.date_range(pd.Timestamp('1988-01') + pd.DateOffset(months=max_input_lag_time+1), '2024-01', freq='MS')

    return first_range.append(second_range)


def generate_split_array(verbose=1):
    start_prediction_months = generate_start_prediction_months()
    split_array = np.empty(np.shape(start_prediction_months), dtype=object)
    
    for i,month in enumerate(start_prediction_months):
        if month in config.TRAIN_MONTHS: split_array[i] = "train"
        if month in config.VAL_MONTHS: split_array[i] = "val"
        if month in config.TEST_MONTHS: split_array[i] = "test"

    if verbose == 2: print_split_stats(split_array)
    
    return split_array, start_prediction_months


def get_device(verbose=1):
    cuda_available = torch.cuda.is_available()
    if verbose >= 1: print(f"CUDA available: {cuda_available}")

    # If available, print the name of the GPU
    if cuda_available and verbose >= 1:
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"Device count: {torch.cuda.device_count()}")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_initialization_seed(seed, verbose=1):
    if verbose >= 2: print(f"Setting random init seed to {seed}")
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
