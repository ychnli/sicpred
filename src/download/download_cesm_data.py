"""
This script downloads monthly CESM2 Large Ensemble variables per ensemble member,
subsets them (time/space/vertical), regrids to an 80x80 South Polar Stereographic (SPS)
target grid using xESMF, and saves one NetCDF file per (variable, member).

Summary:
- Uses the NCAR AWS intake-esm catalog to locate and open CESM2-LENS datasets.
  See https://ncar.github.io/cesm2-le-aws/model_documentation.html for more info
- Builds a list of (variable, member_id) tasks from `DOWNLOAD_SETTINGS`.
- Supports array-job style parallelism. Download and regridding tasks are split among
    workers by interleaving (worker k gets tasks k, k+W, k+2W, ...).
- Preview tasks without downloading using the --dry-run flag.

Usage examples:
Single-worker download (no parallelism):
        python -m src.download.download_cesm_data

Dry-run (prints assignments for a worker without downloading):
        python -m src.download.download_cesm_data --num-workers 8 --worker-id 3 --dry-run

Notes:
- `DOWNLOAD_SETTINGS` controls which variables and which member_id labels to
    process. Member IDs must be the string labels (e.g. 'r5i1301p1f1').
- The script will create output directories under `DOWNLOAD_SETTINGS['save_directory']`
"""

import intake
import numpy as np
import xarray as xr
import xesmf as xe
import time
import os
import pyproj
import argparse
import requests
from src import config_cesm as config
from config import AVAILABLE_CESM_MEMBERS

INPUT_EXPERIMENTS_SUBSET = AVAILABLE_CESM_MEMBERS[0:14]

DOWNLOAD_SETTINGS = {
    # variables to download 
    "vars": ["ICEFRAC", "TEMP", "PSL", "Z3", "TREFHT"],
    # chunk settings
    "chunk": "default",
    # which member ids to download; can be "all" or a list of member ids
    "member_id": {"ICEFRAC": "all", 
                    "TEMP": INPUT_EXPERIMENTS_SUBSET,
                    "PSL": INPUT_EXPERIMENTS_SUBSET,
                    "Z3": INPUT_EXPERIMENTS_SUBSET,
                    "TREFHT": INPUT_EXPERIMENTS_SUBSET
                    },
    # where to save the data
    "save_directory": config.DATA_DIRECTORY
}

# If more variables are added, the following dictionary needs to be updated
# specifically, it contains metadata about which pressure/depth index to subset,
# the variable's long name, and which model component/grid it belongs to
CESM_VAR_ARGS = {
    "ICEFRAC": {
        "p_index": None,
        "save_name": "icefrac",
        "long_name": "Sea ice fraction",
        "grid": "atm"
    },
    "TEMP": {
        "p_index": 0,
        "lat_slice": slice(0, 93),
        "save_name": "sst",
        "long_name": "Sea surface temperature",
        "grid": "ocn"
    },
    "PSL": {
        "p_index": None,
        "save_name": "psl",
        "long_name": "Sea level pressure",
        "grid": "atm"
    },
    "Z3": {
        "p_index": 20,
        "save_name": "geopotential",
        "long_name": "Geopotential height at 500 hPa",
        "grid": "atm"
    },
    "TREFHT": {
        "p_index": None,
        "save_name": "t2m",
        "long_name": "2-meter air temperature",
        "grid": "atm"
    }
}

CATALOG = intake.open_esm_datastore(
    'https://raw.githubusercontent.com/NCAR/cesm2-le-aws/main/intake-catalogs/aws-cesm2-le.json'
)

# CESM's native ocean grid will be loaded at runtime. If missing, download it from ESGF.
CESM_OCEAN_GRID = None

def esgf_search(server="https://esgf-node.llnl.gov/esg-search/search",
                files_type="OPENDAP", local_node=True, project="CMIP6",
                verbose=False, format="application%2Fsolr%2Bjson",
                use_csrf=False, **search):
    """
    Search ESGF using the REST API and return a sorted list of file URLs.

    Author: Unknown
        https://docs.google.com/document/d/1pxz1Kd3JHfFp8vR2JCVBfApbsHmbUQQstifhGNdc6U0/edit?usp=sharing
        API AT: https://github.com/ESGF/esgf.github.io/wiki/ESGF_Search_REST_API#results-pagination
    """
    client = requests.session()
    payload = search.copy()
    payload["project"] = project
    payload["type"] = "File"
    if local_node:
        payload["distrib"] = "false"
    if use_csrf:
        client.get(server)
        if 'csrftoken' in client.cookies:
            csrftoken = client.cookies['csrftoken']
        else:
            csrftoken = client.cookies.get('csrf')
        payload["csrfmiddlewaretoken"] = csrftoken

    payload["format"] = format

    offset = 0
    numFound = 10000
    all_files = []
    files_type = files_type.upper()
    while offset < numFound:
        payload["offset"] = offset
        url_keys = []
        for k in payload:
            url_keys += [f"{k}={payload[k]}"]

        url = f"{server}/?{'&'.join(url_keys)}"
        if verbose:
            print(url)
        r = client.get(url)
        r.raise_for_status()
        resp = r.json()["response"]
        numFound = int(resp["numFound"])
        resp = resp["docs"]
        offset += len(resp)
        for d in resp:
            url = d.get("url", [])
            for f in url:
                sp = f.split("|")
                if sp[-1] == files_type:
                    all_files.append(sp[0].split(".html")[0])
    return sorted(all_files)


def ensure_ocean_grid(verbose=True):
    """Ensure the CESM ocean grid file exists locally; download from ESGF if missing.

    Returns an xarray.Dataset for the ocean grid.
    """
    global CESM_OCEAN_GRID
    save_dir = os.path.join(config.DATA_DIRECTORY, "cesm_data", "grids")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "ocean_grid.nc")

    if os.path.exists(save_path):
        if verbose: print(f"Found existing ocean grid at {save_path}")
        CESM_OCEAN_GRID = xr.open_dataset(save_path)
        return CESM_OCEAN_GRID

    if verbose: print("Ocean grid not found locally; searching ESGF and downloading a CMIP6 file to extract grid...")
    result = esgf_search(activity_id='CMIP', table_id='Omon', variable='thetao', experiment_id='piControl',
                         institution_id="NCAR", source_id="CESM2", member_id='r1i1p1f1')
    if len(result) == 0:
        raise RuntimeError("Could not find a CESM2 ocean grid file on ESGF to download")

    ds = xr.open_dataset(result[0])
    # select one slice and save the result (this should have lat, lon, and bounds)
    ds.isel(time=0, lev=0).to_netcdf(save_path)
    if verbose: print(f"Saved ocean grid to {save_path}")
    CESM_OCEAN_GRID = xr.open_dataset(save_path)
    return CESM_OCEAN_GRID


def retrieve_variable_dataset(catalog, variable, verbose=1):
    """
    Retrieve and merge CESM Large Ensemble historical datasets for a specific variable.
    """

    if verbose >= 1: print(f"Finding {variable}...", end="")
    catalog_subset = catalog.search(variable=variable, frequency='monthly')

    if len(catalog_subset.df) == 0: 
        if verbose >= 1: print(f"did not find any saved data. Skipping...")
        return None 
        
    if len(catalog_subset.df) < 1: 
        if verbose >= 1: print(f"found {len(catalog_subset.df)} saved experiments (unexpected number)")

    dsets = catalog_subset.to_dataset_dict(storage_options={'anon':True})
    if verbose >= 1: print("done! merging datasets...", end="")

    component = catalog_subset.df.component[0]
    CESM_VAR_ARGS[variable]["component"] = component 
    
    cmip_hist_ds = dsets.get(f"{component}.historical.monthly.cmip6", None)
    smbb_hist_ds = dsets.get(f"{component}.historical.monthly.smbb", None)

    if cmip_hist_ds is None and smbb_hist_ds is None:
        raise Exception(f"{variable} seems to be missing historical protocol ensemble members")

    # If both protocol datasets exist, concatenate along member_id; otherwise use whichever exists.
    if cmip_hist_ds is not None and smbb_hist_ds is not None:
        merged_ds = xr.concat([cmip_hist_ds, smbb_hist_ds], dim='member_id')
    elif cmip_hist_ds is not None:
        merged_ds = cmip_hist_ds
    else:
        merged_ds = smbb_hist_ds

    if verbose >= 1: print("done! \n")
    return merged_ds 


def subset_variable_dataset(merged_ds, variable, member_id, chunk="default", time="all", var_args=None):
    """Subset a single member and prep it for regridding."""
    if var_args is None:
        var_args = CESM_VAR_ARGS

    print(f"Subsetting ensemble member {member_id}... ", end="")

    component_grid = var_args[variable]["component"]

    # Select the CESM member_id
    try:
        subset = merged_ds[variable].sel(member_id=member_id)
    except Exception as e:
        print(f"Could not select member {member_id} ({e}). Skipping... ")
        return None
    
    if chunk == "default":
        chunk_settings = {"time": 502}
        # chunk the depths individually so we don't download every depth?
        if var_args[variable]["p_index"] is not None: 
            if component_grid == "ocn": chunk_settings["z_t"] = 1
            elif component_grid == "atm": chunk_settings["lev"] = 1
        subset = subset.chunk(chunk_settings)
    else: 
        subset = subset.chunk(chunk)

    # all the atmospheric variables can be subsetted simply
    if component_grid == "atm":
        subset = subset.sel(lat=slice(-90, -30))
    elif component_grid == "ice":
        if "lat_slice" not in var_args[variable]:
            raise KeyError(f"{variable} requires 'lat_slice' in CESM_VAR_ARGS when component is 'ice'")
        subset = subset.sel(nj=var_args[variable]["lat_slice"])
    else: 
        if "lat_slice" not in var_args[variable]:
            raise KeyError(f"{variable} requires 'lat_slice' in CESM_VAR_ARGS when component is '{component_grid}'")
        subset = subset.sel(nlat=var_args[variable]["lat_slice"])

    # subset z/p dimension, if it exists
    if var_args[variable]["p_index"] is not None: 
        if component_grid == "ocn":
            subset = subset.isel(z_t=var_args[variable]["p_index"])
        else:
            subset = subset.isel(lev=var_args[variable]["p_index"])

    # add latitude/longitude to ocean and ice variables. Both live on the ocean grid
    if component_grid != "atm": 
        if component_grid == "ocn":
            lat_index_name, lon_index_name = "nlat", "nlon"
        elif component_grid == "ice":
            lat_index_name, lon_index_name = "nj", "ni"

        lat = CESM_OCEAN_GRID.lat.sel(nlat=var_args[variable]["lat_slice"]).data
        lon = CESM_OCEAN_GRID.lon.sel(nlat=var_args[variable]["lat_slice"]).data

        subset = subset.assign_coords(lat=([lat_index_name, lon_index_name], lat), lon=([lat_index_name, lon_index_name], lon))
    
    # subset time if specified 
    if time != "all":
        if isinstance(time, slice) or isinstance(time, int): 
            subset = subset.isel(time=time)
        else: return TypeError("time argument must either be 'all', or slice, or int")

    print("done!")
    
    # turn subset into a dataset and rename it 
    subset = subset.to_dataset(name=var_args[variable]["save_name"])

    return subset

def generate_sps_grid(grid_size=80, lat_boundary=-52.5):
    # Define the South Polar Stereographic projection (EPSG:3031)
    proj_south_pole = pyproj.Proj(proj='stere', lat_0=-90, lon_0=0, lat_ts=-70)

    # Define the geographic coordinate system (EPSG:4326)
    proj_geographic = pyproj.Proj(proj='latlong', datum='WGS84')

    # Compute the maximum radius from the South Pole in stereographic coordinates
    _, max_radius = proj_south_pole(0, lat_boundary)

    x = np.linspace(-max_radius, max_radius, grid_size)
    y = np.linspace(-max_radius, max_radius, grid_size)
    X, Y = np.meshgrid(x, y)

    transformer = pyproj.Transformer.from_proj(proj_south_pole, proj_geographic, always_xy=True)
    lon, lat = transformer.transform(X, Y)

    output_grid = xr.Dataset(
        {
            "lat": (["y", "x"], lat),
            "lon": (["y", "x"], lon),
        },
        coords={
            "x": (["x"], x),
            "y": (["y"], y),
        }
    )

    return output_grid


def regrid_variable(ds_to_regrid, input_grid, output_grid):
    start_time = time.time()
    weight_file = f'{config.DATA_DIRECTORY}/cesm_lens/grids/cesm_{input_grid}_to_sps_bilinear_regridding_weights.nc'

    if os.path.exists(weight_file):
        regridder = xe.Regridder(ds_to_regrid, output_grid, 'bilinear', weights=weight_file, 
                                ignore_degenerate=True, reuse_weights=True, periodic=True)
    else:
        regridder = xe.Regridder(ds_to_regrid, output_grid, 'bilinear', filename=weight_file, 
                                ignore_degenerate=True, reuse_weights=False, periodic=True)

    ds_regridded = regridder(ds_to_regrid).load()
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"done! (Time taken: {elapsed_time:.2f} seconds)")
    
    return ds_regridded

def make_save_directories(var_args=CESM_VAR_ARGS, download_settings=DOWNLOAD_SETTINGS, 
                          parent_dir=DOWNLOAD_SETTINGS["save_directory"]):
    """
    Constructs the fire directory structure for saving files

    Args:
    var_args (dict): dictionary containing variable metadata
    download_settings (dict): dictionary containing the download settings
    parent_dir (str): parent directory to save the data

    Returns:
    variable_dirs (dict): dictionary containing the paths to save the variables
    """
    variable_dirs = {}

    for variable in download_settings["vars"]:
        path_name = os.path.join(parent_dir, "cesm_data", var_args[variable]["save_name"])
        os.makedirs(path_name, exist_ok=True)
        variable_dirs[variable] = path_name

    return variable_dirs


def check_if_downloaded(var_args=CESM_VAR_ARGS, download_settings=DOWNLOAD_SETTINGS, 
                        parent_dir=DOWNLOAD_SETTINGS["save_directory"]):
    """
    Given download_settings, find all files that have already been downloaded. 
    Return a modified download_settings only for files that have not been downloaded.

    Args:
    var_args (dict): dictionary containing variable metadata
    download_settings (dict): dictionary containing the download settings
    parent_dir (str): parent directory to save the data

    Returns:
    download_settings_updated (dict): dictionary containing the updated download settings
    """
    download_settings_updated = {
        **download_settings,
        "member_id": {**download_settings.get("member_id", {})},
    }

    for variable in download_settings["vars"]:
        path_name = os.path.join(parent_dir, "cesm_data", var_args[variable]["save_name"])
        files = os.listdir(path_name)
        
        if len(files) > 0:
            # get the specific member ids that have been downloaded
            downloaded_members = []
            for f in files:
                downloaded_members.append(f.split("_")[-1].split(".")[0])
            print(f"Found {len(downloaded_members)} existing downloaded members for requested variable {variable}")

            # remove the downloaded member ids from the download_settings
            download_settings_updated["member_id"][variable] = [m for m in download_settings["member_id"][variable] if str(m) not in downloaded_members]
            
    return download_settings_updated
    
def process_member(variable, merged_ds, input_grid, output_grid, 
                    member, variable_dirs, var_args, chunk):
    """
    Helper function to download, regrid, and save data for a specific ensemble member.
    """
    try:
        subset = subset_variable_dataset(merged_ds, variable, member_id=member, chunk=chunk, var_args=var_args)
        
        if subset is None:
            return

        # Regrid variable
        print(f"Downloading data and regridding for member {member} (this step may take a while)...")
        regridded_subset = regrid_variable(subset, input_grid, output_grid)
        
        # Save the regridded subset
        print(f"Saving member {member}... ", end="")
        file_name = f"{var_args[variable]['save_name']}_{member}.nc"
            
        save_path = os.path.join(variable_dirs[variable], file_name)

        regridded_subset.to_netcdf(save_path)
        regridded_subset.close()
        print("done!")

    except Exception as e:
        print(f"Error processing member {member}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Download, subset, regrid, and save CESM2-LENS variables per ensemble member. "
            "Supports array-job style parallelism via (num_workers, worker_id)."
        )
    )
    parser.add_argument("--num-workers", type=int, default=1, help="Total number of workers in the array job.")
    parser.add_argument("--worker-id", type=int, default=0, help="This worker's id (0-indexed).")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print which (variable, member) pairs would be processed (and by which worker) without downloading.",
    )
    args = parser.parse_args()

    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")
    if not (0 <= args.worker_id < args.num_workers):
        raise ValueError("--worker-id must satisfy 0 <= worker_id < num_workers")

    total_start = time.time()
    timing = {"processed": 0, "skipped_existing": 0, "errors": 0}

    variable_dirs = make_save_directories(
        download_settings=DOWNLOAD_SETTINGS,
        parent_dir=DOWNLOAD_SETTINGS["save_directory"],
        var_args=CESM_VAR_ARGS,
    )

    output_grid = generate_sps_grid()

    # Ensure ocean grid exists (download from ESGF if necessary)
    ensure_ocean_grid()

    download_settings = {
        **DOWNLOAD_SETTINGS,
        "member_id": {**DOWNLOAD_SETTINGS.get("member_id", {})},
    }
    for variable in download_settings["vars"]:
        if download_settings["member_id"].get(variable) == "all":
            download_settings["member_id"][variable] = AVAILABLE_CESM_MEMBERS.copy()

    # check what has already been downloaded and update download settings
    download_settings = check_if_downloaded(
        var_args=CESM_VAR_ARGS,
        download_settings=download_settings,
        parent_dir=DOWNLOAD_SETTINGS["save_directory"],
    )

    # build a list of (variable, member) download/processing tasks
    all_tasks = []
    for variable in download_settings["vars"]:
        for member in download_settings["member_id"][variable]:
            save_path = os.path.join(
                variable_dirs[variable],
                f"{CESM_VAR_ARGS[variable]['save_name']}_{member}.nc",
            )
            if os.path.exists(save_path):
                timing["skipped_existing"] += 1
                continue
            all_tasks.append((variable, member))

    # Evenly split tasks among workers by interleaving 
    # Worker k gets tasks k, k+num_workers, k+2*num_workers, ...
    worker_tasks = all_tasks[args.worker_id :: args.num_workers]

    if args.dry_run:
        print("\n================ CESM download DRY RUN ================")
        print(f"num_workers={args.num_workers} worker_id={args.worker_id}")
        print(f"Total pending tasks (all workers): {len(all_tasks)}")
        print(f"Tasks assigned to this worker:     {len(worker_tasks)}")
        if len(worker_tasks) > 0:
            print("\nFirst 50 tasks for this worker:")
            for (variable, member) in worker_tasks[:50]:
                save_path = os.path.join(
                    variable_dirs[variable],
                    f"{CESM_VAR_ARGS[variable]['save_name']}_{member}.nc",
                )
                print(f"  - worker {args.worker_id}: {variable} member={member} -> {save_path}")
            if len(worker_tasks) > 50:
                print(f"  ... ({len(worker_tasks) - 50} more)")
        print("======================================================\n")
        return

    # cache datasets per variable to avoid repeated catalog reads
    merged_cache = {}
    for (variable, member) in worker_tasks:
        if variable not in merged_cache:
            merged_cache[variable] = retrieve_variable_dataset(catalog=CATALOG, variable=variable)
        merged_ds = merged_cache[variable]
        if merged_ds is None:
            continue

        input_grid = CESM_VAR_ARGS[variable]["grid"]
        per_member_start = time.time()
        try:
            process_member(
                variable,
                merged_ds,
                input_grid,
                output_grid,
                member,
                variable_dirs,
                CESM_VAR_ARGS,
                download_settings["chunk"],
            )
            timing["processed"] += 1
        except Exception:
            timing["errors"] += 1
        finally:
            elapsed = time.time() - per_member_start
            print(f"Member {member} total elapsed: {elapsed/60:.2f} min")

    total_elapsed = time.time() - total_start
    print("\n================ CESM download summary ================")
    print(f"num_workers={args.num_workers} worker_id={args.worker_id}")
    print(f"Total elapsed time: {total_elapsed/60:.2f} min")
    print(f"Processed by this worker: {timing['processed']}")
    print(f"Skipped (existing):       {timing['skipped_existing']}")
    print(f"Errors:                   {timing['errors']}")
    print("======================================================\n")
       

if __name__ == "__main__":
    main()