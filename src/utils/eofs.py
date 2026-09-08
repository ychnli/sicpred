"""EOF analysis helpers for normalized SIC prediction inputs."""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import xarray as xr
import xeofs as xe

from src import config_cesm
from src.experiment_configs import ExperimentConfig, load_config
from src.utils.util_cesm import generate_sps_grid


def compute_eofs(
    data_config: str | ExperimentConfig,
    variables: Sequence[str],
    member_ids: Sequence[str] | None = None,
    n_modes: int = 10,
    **eof_kwargs,
) -> xe.single.EOF:
    """Fit latitude-weighted EOFs to normalized input fields.

    Parameters
    ----------
    data_config
        Named configuration selector (for example, ``"exp1_inputs:input5"``)
        or an already-loaded experiment configuration.
    variables
        Names of normalized variables to include. Passing one name computes a
        univariate EOF; passing multiple names computes a multivariate EOF with
        the same spatial weights applied to every variable.
    member_ids
        Ensemble members to include. By default, use the members common to all
        requested variables.
    n_modes
        Number of EOF modes to compute.
    **eof_kwargs
        Additional keyword arguments passed to :class:`xeofs.single.EOF`.

    Returns
    -------
    xeofs.single.EOF
        The fitted model. Use ``model.components()`` for EOFs and
        ``model.scores()`` for principal components.

    Notes
    -----
    ``time`` and, when present, ``member_id`` are treated as sample dimensions.
    The latitude weights are ``sqrt(cos(latitude))`` because xeofs applies
    supplied weights directly to the data before computing the covariance.
    """
    if isinstance(data_config, str):
        data_config = load_config(data_config)
    elif not isinstance(data_config, ExperimentConfig):
        raise TypeError(
            "data_config must be a configuration selector or ExperimentConfig"
        )

    variables = list(variables)
    if not variables:
        raise ValueError("variables must contain at least one variable name")
    if len(set(variables)) != len(variables):
        raise ValueError("variables must not contain duplicates")
    if n_modes < 1:
        raise ValueError("n_modes must be at least 1")

    normalized_dir = os.path.join(
        config_cesm.PROCESSED_DATA_DIRECTORY,
        "normalized_inputs",
        data_config.data_name,
    )
    arrays = []
    datasets = []
    try:
        for variable in variables:
            path = os.path.join(normalized_dir, f"{variable}_norm.nc")
            dataset = xr.open_dataset(path)
            datasets.append(dataset)
            if variable not in dataset:
                raise ValueError(f"Variable {variable!r} is not present in {path}")
            arrays.append(dataset[variable])

        has_member_dim = ["member_id" in array.dims for array in arrays]
        if any(has_member_dim) and not all(has_member_dim):
            raise ValueError(
                "All requested variables must either have a member_id dimension "
                "or omit it"
            )

        if all(has_member_dim):
            available = list(arrays[0].member_id.values)
            common_members = [
                value
                for value in available
                if all(value in array.member_id.values for array in arrays[1:])
            ]
            selected_members = (
                common_members if member_ids is None else list(member_ids)
            )
            if not selected_members:
                raise ValueError("No ensemble members were selected")
            missing = [
                member for member in selected_members if member not in common_members
            ]
            if missing:
                raise ValueError(
                    f"member_ids are not available for every variable: {missing}"
                )
            arrays = [array.sel(member_id=selected_members) for array in arrays]
        elif member_ids is not None:
            raise ValueError("member_ids were supplied but the data have no member_id")

        arrays = list(xr.align(*arrays, join="inner"))
        if any(array.sizes.get("time", 0) == 0 for array in arrays):
            raise ValueError("The requested variables have no common time values")

        grid = generate_sps_grid(grid_size=arrays[0].sizes["x"]).sel(
            x=arrays[0].x, y=arrays[0].y
        )
        latitude_weights = np.sqrt(
            np.clip(np.cos(np.deg2rad(grid["lat"])), 0.0, None)
        )
        arrays = [
            array.assign_coords(lat=grid["lat"], lon=grid["lon"])
            for array in arrays
        ]
        weights = [latitude_weights] * len(arrays)

        if "use_coslat" in eof_kwargs:
            raise ValueError(
                "use_coslat is managed by compute_eofs for the 2-D projected grid"
            )
        model = xe.single.EOF(
            n_modes=n_modes,
            use_coslat=False,
            **eof_kwargs,
        )
        sample_dims = ["time"]
        if all(has_member_dim):
            sample_dims.append("member_id")
        data = arrays[0] if len(arrays) == 1 else arrays
        fit_weights = weights[0] if len(weights) == 1 else weights
        model.fit(data, dim=sample_dims, weights=fit_weights)
    finally:
        for dataset in datasets:
            dataset.close()

    return model
