import xarray as xr
import config
import util 

def linear_trend(target_month, save_path, linear_years="all"):
    """
    Computes the gridcell-wise linear trend forecast for target_month using linear_years of historical data.

    Params:
        target_month:   pd.datetime object for the month to predict
        save_path:      string 
        linear_years:   int or "all" for all years. If int, must be greater than 1 and less than
                        total years between the target year and the beginning of observations (1987)
    
    Returns:
        prediction:     output map as a xarray Dataset 

    """
    INITIAL_YEAR = pd.to_datetime('1987-11')

    print(f"Computing linear trend forecast for {target_month} using {linear_years} years")
    nsidc_sic = xr.open_dataset(f'{config.DATA_DIRECTORY}/NSIDC/seaice_conc_monthly_all.nc')

    # Select subset of dataset of only the target month 
    subset_target_months = nsidc_sic.siconc.sel(time=nsidc_sic.time.dt.month == target_month.month)

    if linear_years == "all":
        subset_target_months = subset_target_months.sel(time=slice(initial_year, target_month - pd.DateOffset(years=1)))
    else: 
        if type(linear_years) != int:
            raise TypeError("Expected linear_years to be an integer")
        if linear_years == 1: 
            raise ValueError("Cannot make linear trend prediction with only 1 year!")
        if target_month.year - INITIAL_YEAR.year <= linear_years:
            raise ValueError(f"{linear_years} exceeds the total number of years in the timeseries")

        subset_target_months = subset_target_months.sel(time=slice(initial_year - pd.DateOffset(years=linear_years), \
                                                        target_month - pd.DateOffset(years=1)))

    # Mask out land (2.54 and 2.53 flag values)
    land_mask = np.logical_or(nsidc_sic.siconc.isel(time=0) == 2.53, nsidc_sic.siconc.isel(time=0) == 2.54)

    # Mask out open ocean (defined as a gridcell that is always zero)
    all_zeros_mask = np.sum(nsidc_sic.siconc == 0, axis=0) == len(nsidc_sic.time)
    land_and_open_ocean_mask = ~np.logical_or(land_mask.values, all_zeros_mask.values)

    # Compute regression 
    years = subset_target_months.time.dt.year.values
    print(f"Computing linear regression using {linear_years} years...", end=" ")
    reg_result = util.linear_regress_array(years, subset_target_months.values, axis=0, mask=land_and_open_ocean_mask)
    print("done!")

    prediction_npy = reg_result[0] * target_month.year + reg_result[1]

    prediction = xr.Dataset(
        data_vars={
            "siconc": (("ygrid", "xgrid"), prediction_npy)
        }, 
        coords={
            "time": target_month,
            "xgrid": nsidc_sic.xgrid.values,
            "ygrid": nsidc_sic.ygrid.values
        },
        attrs={
            "years_used": linear_years
        }
    )

    prediction.to_netcdf(os.path.join(save_path), f"{target_month.year}-{target_month.month}_{linear_years}_years_linear_forecast.nc")

    return prediction

def anomaly_persistence():
    return None

def unet():
    return None