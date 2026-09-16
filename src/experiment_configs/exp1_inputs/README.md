# Active input configurations

This document summarizes the input variants used by the active experiment pipelines. The source of truth is `src/experiment_configs/exp1_inputs/__init__.py`.

The active variants are `input2`, `input3a`, `input3b`, `input3c`, `input3d`, `input3e`, `input3f`, `input3g`, `input3h_to500`, `input4a`, `input4b`, `input5`, and `input5_noSIC`. The historical `input4` selector remains available for reproducibility. The `input3a_dev`, `input3a_std`, and `input_noise` variants are retained for historical reproducibility but are not called by the current shell pipelines.

## Common design

All active variants contain the following channels:

| Input | Channels | Treatment |
|---|---:|---|
| Sea-ice concentration (`icefrac`) | 12 | The 12 complete months before forecast initialization. Monthly training-set mean is removed, followed by quadratic detrending. Missing values are filled with zero. |
| Initialization month cosine | 1 | A scalar seasonal value broadcast over the full 80 × 80 grid. |
| Initialization month sine | 1 | A scalar seasonal value broadcast over the full 80 × 80 grid. |
| Land mask | 1 | A binary map loaded from the CESM SST grid and broadcast as one channel. |

Each optional physical predictor contributes the six complete months before forecast initialization. Its monthly, grid-cell-specific minimum and maximum are calculated from the training members, it is min-max scaled, and the result is then quadratically detrended. Missing values are filled with zero when model-ready inputs are assembled.

This gives 15 channels for `input2`, 21 for each single-predictor variant, 39 for `input4a`, 33 for revised `input4b`, 63 for `input5`, and 51 for `input5_noSIC`. The no-SIC variant still preprocesses sea-ice concentration as its prediction target, but does not include it among the model inputs. `to500` is CESM ocean `TEMP` (`degC`) selected at native `z_t` index 32: 48,273.671875 cm = 482.737 m, the closest native level to 500 m.

All active variants otherwise share these experiment settings:

- Six monthly sea-ice anomaly targets beginning at the initialization month.
- CESM initialization dates from January 1851 through December 2013.
- Eight training, two validation, and four test ensemble members.
- `UNetRes3` with `n_channels_factor=0.5`.
- Batch size 64, at most 10 epochs, patience 3, and evaluation of the best checkpoint.
- Grid-cell-area-weighted MSE with no target-month weighting.

## Variant matrix

| Selector | Extra physical predictors | Channels | Data artifact | Weight decay |
|---|---|---:|---|---:|
| `exp1_inputs:input2` | None | 15 | `seaice_plus_auxiliary` | `5e-3` |
| `exp1_inputs:input3a` | SST | 21 | `seaice_plus_sst` | `5e-3` |
| `exp1_inputs:input3b` | Sea-level pressure (`psl`) | 21 | `seaice_plus_psl` | `5e-3` |
| `exp1_inputs:input3c` | 500 hPa geopotential height (`z500`) | 21 | `seaice_plus_z500` | `5e-3` |
| `exp1_inputs:input3d` | 2 m air temperature (`t2m`) | 21 | `seaice_plus_t2m` | `5e-3` |
| `exp1_inputs:input3e` | 50 hPa geopotential height (`z50`) | 21 | `seaice_plus_z50` | `5e-3` |
| `exp1_inputs:input3f` | Top-200-m ocean heat content (`ohc200`) | 21 | `seaice_plus_ohc200` | `5e-3` |
| `exp1_inputs:input3g` | Sea-ice thickness (`icethick`) | 21 | `seaice_plus_icethick` | `5e-3` |
| `exp1_inputs:input3h_to500` | Ocean potential temperature at 482.737 m (`to500`) | 21 | `seaice_plus_to500` | `5e-3` |
| `exp1_inputs:input4` | SST, sea-level pressure, 500 hPa geopotential height, and 2 m air temperature | 39 | `seaice_plus_all` | `5e-3` |
| `exp1_inputs:input4a` | z500, z50, psl, and t2m | 39 | `seaice_plus_atmosphere` | `5e-3` |
| `exp1_inputs:input4b` | SST, top-200-m ocean heat content, and `to500` | 33 | `seaice_plus_ocean` | `5e-3` |
| `exp1_inputs:input5` | Sea-ice thickness, SST, top-200-m ocean heat content, `to500`, z500, z50, psl, and t2m | 63 | `seaice_plus_all_variables` | `5e-3` |
| `exp1_inputs:input5_noSIC` | All `input5` predictors except sea-ice concentration | 51 | `all_inputs_no_sic` | `5e-3` |

Experiment 2 copies the complete `input2` input recipe and varies only the number of CESM training members. The active observational and fine-tuning variants in experiment 3 also copy `input2`; they do not use SST or atmospheric predictors.

Bug fixes from previous versions:
1. **Quadratic detrending now fits on the training partition.** The pipeline estimates monthly, grid-cell-specific quadratic coefficients only from training members for member splits or training dates for time splits, then applies them unchanged to held-out data. Existing preprocessed artifacts from before this patch remain contaminated and must not be treated as repaired.

2. **Monthly loss weighting is disabled.** Active exp1 configurations use unit target-month weights, leaving only grid-cell area weighting. Preprocessing retains a training-partition-only monthly-weight artifact for backward compatibility, but these runs do not load it.