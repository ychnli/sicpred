import os 
import xarray as xr
import numpy as np 

from src import config_cesm


var_names = config_cesm.VAR_NAMES

def find_downloaded_vars():
    """
    Finds and verifies the downloaded variables and their ensemble members.
        
    Raises:
        ValueError: If member IDs do not match across variables at any index.

    """

    member_ids = np.empty((len(var_names), 100), dtype=str)
    n_members = []

    for i,variable in enumerate(var_names):
        directory = os.path.join(config_cesm.RAW_DATA_DIRECTORY, variable)
        
        if os.path.exists(directory):
            files = sorted(os.listdir(directory))

            for j,file_name in enumerate(files):
                file_path = os.path.join(directory, file_name)
                ds = xr.open_dataset(file_path)
                member_ids[i,j] = ds["member_id"].values

            print(f"Found {len(files)} ensemble members for {variable}")
            n_members.append(len(files))
        
    # check if member_ids match across variables
    min_members = np.min(n_members)
    for j in range(min_members):
        if not np.all(member_ids[:, j] == member_ids[0, j]):
            raise ValueError(f"Member IDs do not match across variables at index {j}")
    print("All member IDs match across variables")
