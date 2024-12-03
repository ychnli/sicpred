import xarray as xr
import os

RAW_DATA_DIRECTORY = '/scratch/users/yucli/cesm_data'

def main():
    ds = xr.open_dataset(f"{RAW_DATA_DIRECTORY}/normalized_inputs/icefrac_anom.nc")
    smbb_member_ids = ds.member_id[50:].data

    smbb_temp_hist_files = os.listdir("/scratch/users/yucli/cesm_temp_hist_regridded")
    smbb_temp_ssp_files = os.listdir("/scratch/users/yucli/cesm_temp_ssp_regridded")

    smbb_temp_hist_member_ids = []
    smbb_temp_ssp_member_ids = []

    for f in smbb_temp_hist_files:
        smbb_temp_hist_member_ids.append(f.split(".")[0].split("_")[2])
    for f in smbb_temp_ssp_files:
        smbb_temp_ssp_member_ids.append(f.split(".")[0].split("_")[2])

    smbb_temp_member_ids = list(set(smbb_temp_ssp_member_ids) & set(smbb_temp_hist_member_ids))

    for i,member_id in enumerate(smbb_member_ids): 
        if member_id not in smbb_temp_member_ids:
            print(f"Missing {member_id} from SST files")
        else:
            hist_ds = xr.open_dataset(f"/scratch/users/yucli/cesm_temp_hist_regridded/temp_member_{member_id}.nc")
            ssp_ds = xr.open_dataset(f"/scratch/users/yucli/cesm_temp_ssp_regridded/temp_member_{member_id}.nc")

            merged_ds = xr.concat([hist_ds, ssp_ds], dim="time")
            merged_ds.to_netcdf(f'/scratch/users/yucli/cesm_data/temp/temp_member_{50 + i}.nc')
            print(f"temp_member_{50 + i}.nc saved!")

if __name__ == "__main__":
    main()