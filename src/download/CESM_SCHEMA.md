# CESM2-LE Intake schema notes

These results were measured from the live NCAR CESM2-LE AWS Intake catalog with:

```bash
conda activate sicpred_env_conda
python -m src.download.inspect_cesm_schema
```

The inspection utility reads catalog metadata, coordinate arrays, and the static grid. It does not load the full climate-variable arrays.

## Representative variable schemas

| Component | Example | Variable dimensions | Horizontal coordinates | Vertical coordinate | Variable units |
| --- | --- | --- | --- | --- | --- |
| Atmosphere | `Z3` | `member_id, time, lev, lat, lon` | `lat` (`degrees_north`), `lon` (`degrees_east`) | `lev`, 32 hybrid midpoints in `hPa` | `m` |
| Ocean | `TEMP` | `member_id, time, z_t, nlat, nlon` | `TLAT`/`TLONG` from `ocn/static/grid.zarr` | `z_t`, 60 layer midpoints in `centimeters`, positive down | `degC` |
| Sea ice | `hi` | `member_id, time, nj, ni` | Ocean `TLAT`/`TLONG`, with dimensions renamed to `nj`/`ni` | none | `m` |

The sea-ice thickness catalog name is lowercase `hi`, not `HI`. The catalog lists `ice/static/grid.zarr`, but that S3 key does not exist. The ice variable is 384x320 and uses the same horizontal grid as the ocean component, so the ocean static grid supplies its latitude and longitude coordinates.

## Value-based selections used by the downloader

- All outputs use latitude bounds `(-90, -30)` in degrees north.
- Atmospheric data have a one-dimensional monotonic `lat` coordinate and can be selected directly with `.sel(lat=slice(-90, -30))`.
- Ocean and ice data do not carry latitude coordinates in their variable stores. The downloader applies the bounds to the static two-dimensional `TLAT` field, keeps the rectangular row span containing matching cells, masks cells outside the requested bounds, and attaches `TLAT`/`TLONG` as `lat`/`lon` coordinates. For 30 degrees S, the matching native rows are 0 through 92, equivalent to the previous positional `slice(0, 93)`.
- The requested atmospheric 500 hPa level is selected by `lev=500` with `method="nearest"`. The nearest stored hybrid midpoint is 524.6871747 hPa at index 20, preserving the previous `p_index=20` behavior. This is a nominal hybrid level, not pressure interpolation to an exact 500 hPa surface.
- SST uses the `z_t=500` cm midpoint. This is the first ocean layer, bounded by 0 and 1000 cm (0-10 m), preserving the previous `p_index=0` behavior.

## Top-200 m depth average

`ohc200` is the requested thickness-weighted mean potential temperature, not a heat-energy integral. It therefore retains `TEMP` units (`degC`).

The vertical calculation uses `z_w_top` and `z_w_bot` from the ocean static grid, both in centimeters. Model layers 0-19 overlap the interval 0-20,000 cm. The first 19 contributing layers use their full thickness; the final layer begins at 19,182.125 cm and contributes only 817.875 cm, stopping exactly at 200 m. Missing temperatures are excluded and the remaining valid thicknesses are renormalized by `xarray.DataArray.weighted(...).mean("z_t")`, which gives a valid mean in shallow ocean cells and `NaN` where the full selected column is missing.

## Static-grid fields used

- `TLAT`, `TLONG`: two-dimensional T-grid cell centers in degrees north/east.
- `z_t`: layer midpoints in centimeters.
- `z_w_top`, `z_w_bot`: layer interfaces in centimeters.
- `dz`: full layer thickness in centimeters; recorded by the inspection utility for cross-checking the interface-derived overlaps.
