"""Download, derive, regrid, and save monthly CESM2-LE variables.

The pipeline is configured by output variable name. Each output declares the
raw Intake variables it needs and a processor that turns them into a 2-D field
before regridding. Files are saved per output variable and ensemble member.
"""

import argparse
import os
import time

import intake
import numpy as np
import pyproj
import xarray as xr
import xesmf as xe

from src import config_cesm as config


CATALOG_URL = (
    "https://raw.githubusercontent.com/NCAR/cesm2-le-aws/main/"
    "intake-catalogs/aws-cesm2-le.json"
)
INPUT_EXPERIMENTS_SUBSET = config.AVAILABLE_CESM_MEMBERS[:14]

CATALOG = None
CESM_OCEAN_GRID = None


def process_direct(raw_dataset, output_name, settings, ocean_grid=None):
    """Select one raw field and, when requested, one physical vertical level."""
    del ocean_grid
    raw_variable = settings["raw_variables"][0]
    output = raw_dataset[raw_variable]
    vertical = settings.get("vertical_selection")
    if vertical is not None:
        coordinate = vertical["coordinate"]
        if coordinate not in output.coords:
            raise KeyError(
                f"{output_name} requested {coordinate!r}; available coordinates "
                f"are {list(output.coords)}"
            )
        indexer = {coordinate: vertical["value"]}
        method = vertical.get("method")
        output = output.sel(indexer) if method is None else output.sel(indexer, method=method)

    output = output.rename(output_name)
    output.attrs = {
        **output.attrs,
        "long_name": settings["long_name"],
        "source_variables": ",".join(settings["raw_variables"]),
    }
    return output


def process_depth_weighted_mean(raw_dataset, output_name, settings, ocean_grid):
    """Average a field using the overlap of model layers and a depth interval."""
    if ocean_grid is None:
        raise ValueError(f"{output_name} requires the CESM ocean static grid")

    raw_variable = settings["raw_variables"][0]
    depth_coordinate = settings["depth_coordinate"]
    depth_min, depth_max = settings["depth_bounds"]
    field = raw_dataset[raw_variable]

    required = {depth_coordinate, "z_w_top", "z_w_bot"}
    missing = required.difference(ocean_grid.variables)
    if missing:
        raise KeyError(f"Ocean grid is missing {sorted(missing)}")

    centers = ocean_grid[depth_coordinate].values
    layer_top = xr.DataArray(
        ocean_grid["z_w_top"].values,
        dims=(depth_coordinate,),
        coords={depth_coordinate: centers},
    )
    layer_bottom = xr.DataArray(
        ocean_grid["z_w_bot"].values,
        dims=(depth_coordinate,),
        coords={depth_coordinate: centers},
    )
    clipped_top = xr.where(layer_top < depth_min, depth_min, layer_top)
    clipped_bottom = xr.where(layer_bottom > depth_max, depth_max, layer_bottom)
    weights = (clipped_bottom - clipped_top).clip(min=0)
    weights = weights.where(weights > 0, drop=True)
    if weights.size == 0:
        raise ValueError(
            f"No {depth_coordinate} layers overlap {settings['depth_bounds']}"
        )

    field = field.sel({depth_coordinate: weights[depth_coordinate]})
    output = field.weighted(weights).mean(depth_coordinate).rename(output_name)
    depth_units = ocean_grid[depth_coordinate].attrs.get("units", "unknown")
    output.attrs = {
        "long_name": settings["long_name"],
        "units": field.attrs.get("units", ""),
        "source_variables": ",".join(settings["raw_variables"]),
        "depth_bounds": f"{depth_min} to {depth_max} {depth_units}",
        "processing": "layer-overlap-weighted vertical mean",
    }
    return output


# Keys are output/save names. Raw catalog variables are declared explicitly so
# multiple outputs can share a raw download (TEMP is shared by sst and ohc200).
CESM_VAR_ARGS = {
    "icefrac": {
        "raw_variables": ("ICEFRAC",),
        "processor": process_direct,
        "latitude_bounds": (-90.0, -30.0),
        "long_name": "Sea ice fraction",
        "grid": "atm",
    },
    "sst": {
        "raw_variables": ("TEMP",),
        "processor": process_direct,
        "vertical_selection": {
            "coordinate": "z_t",
            "value": 500.0,
            "units": "centimeters",
        },
        "latitude_bounds": (-90.0, -30.0),
        "long_name": "Sea surface temperature",
        "grid": "ocn",
    },
    "ohc200": {
        "raw_variables": ("TEMP",),
        "processor": process_depth_weighted_mean,
        "depth_coordinate": "z_t",
        "depth_bounds": (0.0, 20_000.0),
        "depth_units": "centimeters",
        "latitude_bounds": (-90.0, -30.0),
        "long_name": "Top 200 m depth-averaged ocean potential temperature",
        "grid": "ocn",
    },
    "psl": {
        "raw_variables": ("PSL",),
        "processor": process_direct,
        "latitude_bounds": (-90.0, -30.0),
        "long_name": "Sea level pressure",
        "grid": "atm",
    },
    "geopotential": {
        "raw_variables": ("Z3",),
        "processor": process_direct,
        "vertical_selection": {
            "coordinate": "lev",
            "value": 500.0,
            "units": "hPa",
            "method": "nearest",
        },
        "latitude_bounds": (-90.0, -30.0),
        "long_name": "Geopotential height near 500 hPa",
        "grid": "atm",
    },
    "t2m": {
        "raw_variables": ("TREFHT",),
        "processor": process_direct,
        "latitude_bounds": (-90.0, -30.0),
        "long_name": "2-meter air temperature",
        "grid": "atm",
    },
}

DOWNLOAD_SETTINGS = {
    "vars": ["icefrac", "sst", "ohc200", "psl", "geopotential", "t2m"],
    "chunk": "default",
    "member_id": {
        "icefrac": "all",
        "sst": INPUT_EXPERIMENTS_SUBSET,
        "ohc200": INPUT_EXPERIMENTS_SUBSET,
        "psl": INPUT_EXPERIMENTS_SUBSET,
        "geopotential": INPUT_EXPERIMENTS_SUBSET,
        "t2m": INPUT_EXPERIMENTS_SUBSET,
    },
    "save_directory": config.DATA_DIRECTORY,
}


def get_catalog():
    """Open the Intake catalog once, on first use."""
    global CATALOG
    if CATALOG is None:
        CATALOG = intake.open_esm_datastore(CATALOG_URL)
    return CATALOG


def load_ocean_grid(catalog=None, verbose=True):
    """Load horizontal coordinates and vertical geometry from the static grid."""
    global CESM_OCEAN_GRID
    if CESM_OCEAN_GRID is not None:
        return CESM_OCEAN_GRID

    catalog = get_catalog() if catalog is None else catalog
    if verbose:
        print("Loading CESM ocean static grid... ", end="")
    subset = catalog.search(
        component="ocn",
        frequency="static",
        experiment="historical",
        forcing_variant="cmip6",
    )
    datasets = subset.to_dataset_dict(storage_options={"anon": True})
    if len(datasets) != 1:
        raise RuntimeError(f"Expected one ocean grid, found {sorted(datasets)}")

    source = next(iter(datasets.values()))
    required = ("TLAT", "TLONG", "z_t", "z_w_top", "z_w_bot", "dz")
    missing = set(required).difference(source.variables)
    if missing:
        raise KeyError(f"Ocean grid is missing {sorted(missing)}")
    CESM_OCEAN_GRID = xr.Dataset({name: source[name] for name in required}).load()
    if verbose:
        print("done!")
    return CESM_OCEAN_GRID


def retrieve_variable_dataset(catalog, variable, verbose=1):
    """Retrieve and merge historical forcing variants for one raw variable."""
    if verbose:
        print(f"Finding raw variable {variable}...", end="")
    subset = catalog.search(
        variable=variable,
        frequency="monthly",
        experiment="historical",
    )
    if len(subset.df) == 0:
        if verbose:
            print("not found")
        return None

    components = subset.df["component"].dropna().unique()
    if len(components) != 1:
        raise RuntimeError(
            f"Expected {variable} on one component, found {components.tolist()}"
        )
    component = components[0]
    datasets = subset.to_dataset_dict(storage_options={"anon": True})
    cmip = datasets.get(f"{component}.historical.monthly.cmip6")
    smbb = datasets.get(f"{component}.historical.monthly.smbb")
    if cmip is None and smbb is None:
        raise RuntimeError(f"{variable} has no historical datasets")
    if cmip is not None and smbb is not None:
        merged = xr.concat([cmip, smbb], dim="member_id")
    else:
        merged = cmip if cmip is not None else smbb
    if verbose:
        print("done!")
    return merged


def _select_latitude(dataset, component_grid, latitude_bounds, ocean_grid):
    """Subset rectilinear or curvilinear grids using latitude values."""
    latitude_min, latitude_max = sorted(latitude_bounds)
    if component_grid == "atm":
        return dataset.sel(lat=slice(latitude_min, latitude_max))

    if ocean_grid is None:
        raise ValueError(f"The {component_grid} component requires the ocean grid")
    if component_grid == "ocn":
        row_dim, column_dim = "nlat", "nlon"
        latitude = ocean_grid["TLAT"]
        longitude = ocean_grid["TLONG"]
    elif component_grid == "ice":
        row_dim, column_dim = "nj", "ni"
        rename_dims = {"nlat": row_dim, "nlon": column_dim}
        latitude = ocean_grid["TLAT"].rename(rename_dims)
        longitude = ocean_grid["TLONG"].rename(rename_dims)
    else:
        raise ValueError(f"Unsupported CESM component: {component_grid}")

    latitude_mask = (latitude >= latitude_min) & (latitude <= latitude_max)
    matching_rows = np.flatnonzero(latitude_mask.any(column_dim).values)
    if matching_rows.size == 0:
        raise ValueError(f"No grid cells lie within {latitude_bounds}")
    row_slice = slice(int(matching_rows[0]), int(matching_rows[-1]) + 1)
    dataset = dataset.isel({row_dim: row_slice})
    latitude = latitude.isel({row_dim: row_slice})
    longitude = longitude.isel({row_dim: row_slice})
    latitude_mask = latitude_mask.isel({row_dim: row_slice})

    for name in dataset.data_vars:
        if {row_dim, column_dim}.issubset(dataset[name].dims):
            dataset[name] = dataset[name].where(latitude_mask)
    return dataset.assign_coords(
        lat=((row_dim, column_dim), latitude.values),
        lon=((row_dim, column_dim), longitude.values),
    )


def subset_variable_dataset(
    raw_datasets,
    derived_variable,
    member_id,
    chunk="default",
    time_selection="all",
    var_args=None,
    ocean_grid=None,
):
    """Select one member, subset coordinates, and create a derived output."""
    var_args = CESM_VAR_ARGS if var_args is None else var_args
    settings = var_args[derived_variable]
    print(f"Subsetting {derived_variable} for member {member_id}... ", end="")

    selected = {}
    for raw_variable in settings["raw_variables"]:
        try:
            selected[raw_variable] = raw_datasets[raw_variable][raw_variable].sel(
                member_id=member_id
            )
        except (KeyError, ValueError) as error:
            print(f"could not select member ({error})")
            return None
    member_dataset = xr.Dataset(selected)
    member_dataset = _select_latitude(
        member_dataset,
        settings["grid"],
        settings["latitude_bounds"],
        ocean_grid,
    )

    if time_selection != "all":
        if not isinstance(time_selection, (slice, int)):
            raise TypeError("time_selection must be 'all', a slice, or an integer")
        member_dataset = member_dataset.isel(time=time_selection)

    output = settings["processor"](
        member_dataset,
        derived_variable,
        settings,
        ocean_grid,
    )
    if chunk == "default" and "time" in output.dims:
        output = output.chunk({"time": 502})
    elif chunk != "default":
        output = output.chunk(chunk)
    print("done!")
    return output.to_dataset(name=derived_variable)


def generate_sps_grid(grid_size=80, lat_boundary=-52.5):
    """Construct the target South Polar Stereographic grid."""
    south_pole = pyproj.Proj(proj="stere", lat_0=-90, lon_0=0, lat_ts=-70)
    geographic = pyproj.Proj(proj="latlong", datum="WGS84")
    _, max_radius = south_pole(0, lat_boundary)
    x = np.linspace(-max_radius, max_radius, grid_size)
    y = np.linspace(-max_radius, max_radius, grid_size)
    x_mesh, y_mesh = np.meshgrid(x, y)
    transformer = pyproj.Transformer.from_proj(south_pole, geographic, always_xy=True)
    lon, lat = transformer.transform(x_mesh, y_mesh)
    return xr.Dataset(
        {"lat": (["y", "x"], lat), "lon": (["y", "x"], lon)},
        coords={"x": (["x"], x), "y": (["y"], y)},
    )


def regrid_variable(dataset, input_grid, output_grid):
    """Regrid one derived dataset to the common SPS grid."""
    start_time = time.time()
    weights_dir = os.path.join(config.DATA_DIRECTORY, "cesm_lens", "grids")
    os.makedirs(weights_dir, exist_ok=True)
    weight_file = os.path.join(
        weights_dir, f"cesm_{input_grid}_to_sps_bilinear_regridding_weights.nc"
    )
    kwargs = {
        "ignore_degenerate": True,
        "periodic": True,
    }
    if os.path.exists(weight_file):
        kwargs.update(weights=weight_file, reuse_weights=True)
    else:
        kwargs.update(filename=weight_file, reuse_weights=False)
    regridder = xe.Regridder(dataset, output_grid, "bilinear", **kwargs)
    regridded = regridder(dataset).load()
    print(f"done! (Time taken: {time.time() - start_time:.2f} seconds)")
    return regridded


def make_save_directories(
    var_args=CESM_VAR_ARGS,
    download_settings=DOWNLOAD_SETTINGS,
    parent_dir=DOWNLOAD_SETTINGS["save_directory"],
):
    """Create output directories keyed by derived variable name."""
    directories = {}
    for output_name in download_settings["vars"]:
        if output_name not in var_args:
            raise KeyError(f"Unknown derived variable: {output_name}")
        path = os.path.join(parent_dir, "cesm_data", output_name)
        os.makedirs(path, exist_ok=True)
        directories[output_name] = path
    return directories


def check_if_downloaded(
    var_args=CESM_VAR_ARGS,
    download_settings=DOWNLOAD_SETTINGS,
    parent_dir=DOWNLOAD_SETTINGS["save_directory"],
):
    """Remove already-saved output/member pairs from requested settings."""
    del var_args
    updated = {
        **download_settings,
        "member_id": {**download_settings.get("member_id", {})},
    }
    for output_name in download_settings["vars"]:
        directory = os.path.join(parent_dir, "cesm_data", output_name)
        prefix = f"{output_name}_"
        downloaded = [
            filename[len(prefix) : -3]
            for filename in os.listdir(directory)
            if filename.startswith(prefix) and filename.endswith(".nc")
        ]
        if downloaded:
            print(f"Found {len(downloaded)} existing members for {output_name}")
            updated["member_id"][output_name] = [
                member
                for member in download_settings["member_id"][output_name]
                if str(member) not in downloaded
            ]
    return updated


def process_member(
    derived_variable,
    raw_datasets,
    output_grid,
    member,
    variable_dirs,
    var_args,
    chunk,
    ocean_grid,
):
    """Derive, regrid, and save one output/member pair."""
    subset = subset_variable_dataset(
        raw_datasets,
        derived_variable,
        member,
        chunk=chunk,
        var_args=var_args,
        ocean_grid=ocean_grid,
    )
    if subset is None:
        return False
    print(f"Downloading and regridding {derived_variable} for member {member}...")
    regridded = regrid_variable(
        subset, var_args[derived_variable]["grid"], output_grid
    )
    save_path = os.path.join(
        variable_dirs[derived_variable], f"{derived_variable}_{member}.nc"
    )
    print(f"Saving {save_path}... ", end="")
    try:
        regridded.to_netcdf(save_path)
    finally:
        regridded.close()
        subset.close()
    print("done!")
    return True


def _expanded_download_settings(download_settings):
    """Copy settings and expand the special ``all`` member selector."""
    expanded = {
        **download_settings,
        "member_id": {**download_settings.get("member_id", {})},
    }
    for output_name in expanded["vars"]:
        if expanded["member_id"].get(output_name) == "all":
            expanded["member_id"][output_name] = config.AVAILABLE_CESM_MEMBERS.copy()
    return expanded


def _pending_tasks(download_settings, variable_dirs):
    """Return unsaved ``(output_name, member)`` tasks and a skipped count."""
    tasks = []
    skipped = 0
    for output_name in download_settings["vars"]:
        for member in download_settings["member_id"][output_name]:
            path = os.path.join(variable_dirs[output_name], f"{output_name}_{member}.nc")
            if os.path.exists(path):
                skipped += 1
            else:
                tasks.append((output_name, member))
    return tasks, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Download, derive, regrid, and save CESM2-LE variables."
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")
    if not 0 <= args.worker_id < args.num_workers:
        raise ValueError("--worker-id must satisfy 0 <= worker-id < num-workers")

    total_start = time.time()
    timing = {"processed": 0, "skipped_existing": 0, "errors": 0}
    variable_dirs = make_save_directories()
    settings = _expanded_download_settings(DOWNLOAD_SETTINGS)
    settings = check_if_downloaded(
        download_settings=settings,
        parent_dir=DOWNLOAD_SETTINGS["save_directory"],
    )
    all_tasks, timing["skipped_existing"] = _pending_tasks(settings, variable_dirs)
    worker_tasks = all_tasks[args.worker_id :: args.num_workers]

    if args.dry_run:
        print("\n================ CESM download DRY RUN ================")
        print(f"num_workers={args.num_workers} worker_id={args.worker_id}")
        print(f"Total pending tasks (all workers): {len(all_tasks)}")
        print(f"Tasks assigned to this worker:     {len(worker_tasks)}")
        for output_name, member in worker_tasks[:50]:
            raw = ",".join(CESM_VAR_ARGS[output_name]["raw_variables"])
            path = os.path.join(variable_dirs[output_name], f"{output_name}_{member}.nc")
            print(f"  - {output_name} (raw: {raw}) member={member} -> {path}")
        if len(worker_tasks) > 50:
            print(f"  ... ({len(worker_tasks) - 50} more)")
        print("======================================================\n")
        return

    catalog = get_catalog()
    output_grid = generate_sps_grid()
    needs_ocean_grid = any(
        CESM_VAR_ARGS[output_name]["grid"] != "atm"
        for output_name, _ in worker_tasks
    )
    ocean_grid = load_ocean_grid(catalog) if needs_ocean_grid else None
    raw_cache = {}
    try:
        for output_name, member in worker_tasks:
            task_start = time.time()
            try:
                raw_datasets = {}
                for raw_variable in CESM_VAR_ARGS[output_name]["raw_variables"]:
                    if raw_variable not in raw_cache:
                        raw_cache[raw_variable] = retrieve_variable_dataset(
                            catalog, raw_variable
                        )
                    if raw_cache[raw_variable] is None:
                        raise RuntimeError(f"Could not retrieve {raw_variable}")
                    raw_datasets[raw_variable] = raw_cache[raw_variable]
                if process_member(
                    output_name,
                    raw_datasets,
                    output_grid,
                    member,
                    variable_dirs,
                    CESM_VAR_ARGS,
                    settings["chunk"],
                    ocean_grid,
                ):
                    timing["processed"] += 1
                else:
                    timing["errors"] += 1
            except Exception as error:
                timing["errors"] += 1
                print(f"Error processing {output_name}, member {member}: {error}")
            finally:
                print(f"Task elapsed: {(time.time() - task_start) / 60:.2f} min")
    finally:
        for dataset in raw_cache.values():
            if dataset is not None:
                dataset.close()

    print("\n================ CESM download summary ================")
    print(f"num_workers={args.num_workers} worker_id={args.worker_id}")
    print(f"Total elapsed time: {(time.time() - total_start) / 60:.2f} min")
    print(f"Processed by this worker: {timing['processed']}")
    print(f"Skipped (existing):       {timing['skipped_existing']}")
    print(f"Errors:                   {timing['errors']}")
    print("======================================================\n")


if __name__ == "__main__":
    main()
