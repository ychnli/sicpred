import intake
import xarray as xr
import os

from src import download_cesm_data
from src.download_cesm_data import var_args 


DATA_DIRECTORY = '/oak/stanford/groups/earlew/yuchen'

CATALOG = intake.open_esm_datastore(
    'https://raw.githubusercontent.com/NCAR/cesm2-le-aws/main/intake-catalogs/aws-cesm2-le.json'
)

CESM_OCEAN_GRID = xr.open_dataset(f"{DATA_DIRECTORY}/cesm_lens/grids/ocean_grid.nc")


def retrieve_dataset(catalog, variable, verbose=1):
    """
    
    """

    if verbose >= 1: print(f"Finding {variable}...", end="")
    catalog_subset = CATALOG.search(variable=variable, frequency='monthly')

    if len(catalog_subset.df) == 0: 
        if verbose >= 1: print(f"did not find any saved data. Skipping...")
        return None 
        
    if len(catalog_subset.df) != 4: 
        if verbose >= 1: print(f"only found {len(catalog_subset.df)} saved experiments instead of expected (4)")

    dsets = catalog_subset.to_dataset_dict(storage_options={'anon':True})

    # get the model component and save it to var_args dict 
    component = catalog_subset.df.component[0]
    var_args[variable]["component"] = component 
    
    smbb_ssp_ds = dsets.get(f"{component}.ssp370.monthly.smbb", None)

    return smbb_ssp_ds 


def main():
    input_grid = "ocn"
    output_grid = download_cesm_data.generate_sps_grid()
    
    merged_ds = retrieve_dataset(catalog=CATALOG, variable="TEMP")
    
    variable_dirs = {}
    variable_dirs["TEMP"] = "/scratch/users/yucli/cesm_temp_ssp_regridded"

    for i, member_id in enumerate(merged_ds.member_id.data):
        save_path = os.path.join("/scratch/users/yucli/cesm_temp_ssp_regridded", f"temp_member_{member_id}.nc")
        if os.path.exists(save_path):
            print(f"Already found existing {save_path}, skipping")
            continue

        download_cesm_data.process_member("TEMP", merged_ds, input_grid, output_grid, i,
            variable_dirs, var_args, chunk="default", save_name="id_code", save_name_id=member_id)
    

if __name__ == "__main__":
    main()