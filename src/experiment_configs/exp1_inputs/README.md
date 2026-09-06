# Active input configurations

This document summarizes the input variants used by the active experiment pipelines. The source of truth is `src/experiment_configs/exp1_inputs/__init__.py`.

The active variants are `input2`, `input3a`, `input3b`, `input3c`, `input3d`, and `input4`. The `input3a_dev`, `input3a_std`, and `input_noise` variants are retained for historical reproducibility but are not called by the current shell pipelines.

## Common design

All active variants contain the following channels:

| Input | Channels | Treatment |
|---|---:|---|
| Sea-ice concentration (`icefrac`) | 12 | The 12 complete months before forecast initialization. Monthly training-set mean is removed, followed by quadratic detrending. Missing values are filled with zero. |
| Initialization month cosine | 1 | A scalar seasonal value broadcast over the full 80 × 80 grid. |
| Initialization month sine | 1 | A scalar seasonal value broadcast over the full 80 × 80 grid. |
| Land mask | 1 | A binary map loaded from the CESM SST grid and broadcast as one channel. |

Each optional physical predictor contributes the six complete months before forecast initialization. Its monthly, grid-cell-specific minimum and maximum are calculated from the training members, it is min-max scaled, and the result is then quadratically detrended. Missing values are filled with zero when model-ready inputs are assembled.

This gives 15 channels for `input2`, 21 for each single-predictor variant, and 39 for `input4`.

All six variants otherwise share these experiment settings:

- Six monthly sea-ice anomaly targets beginning at the initialization month.
- CESM initialization dates from January 1851 through December 2013.
- Eight training, two validation, and four test ensemble members.
- `UNetRes3` with `n_channels_factor=0.5`.
- Batch size 64, at most 10 epochs, patience 3, and evaluation of the best checkpoint.
- Area- and target-month-weighted MSE in the current training implementation.

## Variant matrix

| Selector | Extra physical predictors | Channels | Data artifact | Weight decay |
|---|---|---:|---|---:|
| `exp1_inputs:input2` | None | 15 | `seaice_plus_auxiliary` | `1e-3` |
| `exp1_inputs:input3a` | SST | 21 | `seaice_plus_sst` | `5e-3` |
| `exp1_inputs:input3b` | Sea-level pressure (`psl`) | 21 | `seaice_plus_psl` | `5e-3` |
| `exp1_inputs:input3c` | 500 hPa geopotential height (`geopotential`) | 21 | `seaice_plus_z500` | `5e-3` |
| `exp1_inputs:input3d` | 2 m air temperature (`t2m`) | 21 | `seaice_plus_t2m` | `5e-3` |
| `exp1_inputs:input4` | SST, sea-level pressure, 500 hPa geopotential height, and 2 m air temperature | 39 | `seaice_plus_all` | `5e-3` |

Experiment 2 copies the complete `input2` input recipe and varies only the number of CESM training members. The active observational and fine-tuning variants in experiment 3 also copy `input2`; they do not use SST or atmospheric predictors.

## Design choices worth revisiting

### High priority

1. **The physical-variable names are inconsistent across configuration, registry, and downloading.** The active configs request `psl` and `geopotential`. The current `ALL_VAR_NAMES` registry omits `psl` and contains both `geopotential` and `z500`, while the current downloader writes 500 hPa geopotential height as `z500`. `merge_data_by_member()` and input loading iterate over `ALL_VAR_NAMES`, so a fresh preprocessing run can skip `psl` and can look for geopotential under a different name from the downloader. This could make `input3b`, `input3c`, and `input4` fail preprocessing or contain fewer channels than the model expects.

2. **Quadratic detrending is fitted using more than the training partition.** Monthly scaling statistics are correctly calculated from the training subset, but `detrend_quadratic()` is subsequently fitted to the full selected array. For CESM experiments that includes validation and test members; for time-split observational experiments it includes future validation and test dates. This is data leakage and will also matter for future rolling-origin cross-validation.

3. **Monthly loss weights use the full normalized dataset.** `calculate_monthly_weights()` averages `icefrac_norm.nc` over every available time and member instead of selecting the training partition. Validation/test information therefore influences the training objective.

4. **The input ablation changes weight decay as well as inputs.** `input2` uses `1e-3`, whereas all variants with additional predictors use `5e-3`. Comparisons against `input2` therefore do not isolate the value of the added feature set.

### Medium priority

5. **The per-variable `land_mask` setting is currently inert.** Values such as `icefrac["land_mask"]` and `sst["land_mask"]` are defined but never read during normalization or input assembly. NaNs are always replaced by zero, and the only explicit land information comes from the separate auxiliary land-mask channel. The unused setting can mislead a reader into believing different masking behavior is being applied.

6. **Quadratic detrending is an implicit global default.** Every normalized physical variable is detrended because `normalize_data(..., detrend=True)` is not overridden by the experiment config. That includes SST and all atmospheric inputs. After detrending, a variable described as min-max scaled is no longer necessarily bounded by `[0, 1]`. This choice should either be made explicit in the config or justified as a fixed preprocessing invariant.

7. **Min-max scaling has no zero-range guard.** At grid cells/months where the training minimum equals the maximum, normalization divides by zero. Later `fillna(0)` handles NaNs during assembly, but it does not explicitly handle positive or negative infinity.

8. **The land-mask orientation relies on an unexplained transpose.** Input construction reshapes and transposes the mask with the comment “for some reason, x and y get switched.” This deserves a coordinate-based assertion or visual test, especially because the same mask recipe is inherited by observational experiments.

### Lower priority / intentional but worth remembering

9. **Seasonal inputs are constant images.** Sine and cosine values are repeated over the entire spatial grid. This is a reasonable way to supply seasonality to a convolutional model, though it uses two full image channels for two scalars.

10. **All physical inputs stop one month before initialization.** For a forecast initialized in month `t`, inputs end at `t-1` and targets begin at `t`. This avoids contemporaneous target information, but the convention should remain explicit when comparing with systems whose initialization data include month `t`.

11. **The target lead coordinate is hard-coded to six values during preprocessing.** Although `max_lead_months` is configurable, target construction assigns `np.arange(1, 7)`. Any future experiment using a different forecast horizon would produce inconsistent coordinates.

12. **Anomaly outputs are constrained by `tanh`.** `UNetRes3` maps anomaly predictions to `[-1, 1]`. Sea-ice concentration anomalies are naturally limited to roughly this range before detrending, but the quadratic transformation can move values outside it. It is worth checking the processed target distribution for clipping pressure.
