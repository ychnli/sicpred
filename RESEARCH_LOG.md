# Research Log

This is a running log for experiment status, decisions, and findings. It is intended for both researchers and agents.

## Logging convention

- Add new entries in chronological order under a heading formatted as `YYYY-MM-DD HH:MM TZ`.
- Use the local timezone and include it explicitly.
- Keep entries concise, and record affected experiments, current status, blockers or caveats, and next actions when known.
- Do not silently revise an older entry when the status changes; append a new timestamped entry instead.

## 2026-09-09 20:36 PDT

### Experiment 1: input configurations (`exp1_inputs`)

- **minor data leakage bug** All current experiment results are contaminated by a data-leakage bug since quadratic detrending was fitted using validation and test data in addition to training data. However, this generally has a small effect on the data and will probably have only a minor effect on modeling results, particularly differences between input configurations. We should patch the bug for all new data preprocessing going forward, but repair of the older runs is a lower priority for now.
- **Sea-ice thickness:** `icethick` has very small training-set min-max ranges at some grid cells/months. Min-max normalization can therefore amplify values severely (and has no near-zero-range safeguard), causing inputs to blow up. This directly affects `input3g` and also affects the combined configurations `input5` and `input5_noSIC`.
- **Configuration reference:** See `src/experiment_configs/exp1_inputs/README.md` for the experiment design and known preprocessing concerns. It predates some newer input configurations, so verify the active selectors against `src/experiment_configs/exp1_inputs/__init__.py` before dispatching reruns.

Todo:
1. first, fix the detrending data leakage issue by confining the detrending fit to the training set (this should come before any new tasks that require data preprocessing)

2. we need to come up with a way to clip the icethick data. There is a similar issue of small training-set ranges that lead to high normalized values in extrapolation for sst, but it is much less bad (max magnitudes in test and val end up being ~8.0). One extreme could be to just clip everything to [-1, 1], but I think there are reasons not to do this, e.g., it would introduce input vectors that are not consistent over the physical variables. Another idea is to set an arbitrary clipping range (e.g., max extrapolated magnitude of 10). Before committing to any one of these solutions, it would be good to develop a bit of analysis around which grid points/months have tiny training range. Add on to the icethick triage notebook for this analysis; do not preprocess new data or train new models before approval.

New input experiments:
3. new input to500: we should add subsurface ocean potential temperature at 500m (or the closest model depth level) as an input for the experiments (run download and regrid). Create a new configuration input3h_to500 which includes 6 months of to500 and 12 months of SIC. Modify the existing SIC+ocean configuration input4b and overwrite the previous preprocessed data files. Run a GPU job to train, evaluate, and compute diagnostics+bootstrap for input3h; there are already existing checkpoints for input4b, but this previous training run didn't include to500. Make a copy of those checkpoints, and retrain input4b including to500 as part of the ocean inputs. 

4. new inputs surface wind: a previous intuition is that the sea level pressure input should be mostly synonymous with surface winds, since monthly averaged winds will be approximately geostrophic. It would be good to confirm this. Create a new configuration input3i_sfcwinds and fetch zonal and meridional winds (I'm not sure what variable this is in the intake catalog, so you will have to search the catalog first; see sicpred/src/download/inspect_cesm_schema.py for an example for how to do this). Preprocess the data, and queue a GPU job to train this configuration. Produce model predictions using evaluate, and diagnostics, and bootstrap intervals against input2 similar to all of the other configurations.

5. Once (3) and (4) are done, update the figure in exp1_figures_extended.ipynb to include *all* finished input configuration runs.

### Experiment 2:
Todo:
1. currently, our smallest data configuration consists of a single CESM2-LE ensemble member which roughly spans 1850-2014. However, it would be good to conduct a set of experiments that mimic the data volume of reanalysis, which spans roughly 1979-2025, but do it within CESM2-LE, so we can directly compare it with our data scaling results. Specifically, use the temporal data train/val/test split as in exp3. Use the same training settings as exp3, but use CESM2-LE data. Train 5 neural network seeds for each *testing* ensemble member in exp2 (so that we can also compare the results with models trained with much more data volume); note there is no data leakage because our data split is temporal rather than by member. Since there are 4 CESM ensemble members, you should train 20 models total. 


Other notes (for future reference, do not queue new agents to work on this right now)
- input2 unintentionally used a different regularization setting than the other input configurations. We should repair this in our resubmission 
- eventually need to repair all runs which have data leakage (which includes exp2 and exp3)
- if we are re-running exp2, it would be good to use the same held out testing members as exp1 -- this way we can do a apples-to-apples comparison between data scaling results and variable importance results

## 2026-09-09 22:42 PDT

### Task 1: training-only detrending patch

- **Status:** Complete. `src/utils/util_cesm.py` now fits monthly quadratic detrending coefficients strictly on the training partition and applies them unchanged to validation/test data: training members for ensemble-member splits (`exp1`/`exp2`) and training dates for time splits (`exp3`). Focused regression coverage was added in `tests/test_util_cesm.py`.
- **Validation:** The full test suite passed (43 tests), `python -m compileall src` passed, and production output matched `experiments/exp1_inputs/detrending_triage/generate_and_compare.py` on a representative CESM subset.
- **Artifacts:** No preprocessing or model artifacts were regenerated, and no Slurm/GPU jobs were launched. Existing normalized data, data pairs, checkpoints, predictions, and diagnostics remain unchanged.

## 2026-09-09 23:51 PDT

### Task 2: Experiment 2 reanalysis-volume CESM analogue

- **Status:** Submitted. Four isolated, contiguous time-split CESM configurations use members `r2i1251p1f1`, `r2i1281p1f1`, `r2i1301p1f1`, and `r3i1041p1f1` with train/validation/test periods 1968-2000, 2001-2006, and 2007-2013. They use the non-finetune exp3 optimization settings and five neural seeds per member (20 models).
- **Jobs:** CPU preprocessing array `42700825` (4 tasks) is pending; GPU A100/H100 training array `42700827` (20 tasks) will run after it succeeds; GPU evaluation/diagnostics array `42700828` (4 tasks) will run after all training tasks succeed. Existing `vol1`-`vol4` runs were not touched.
- **Validation:** 52 tests passed; `python -m compileall src`, `bash -n`, `sbatch --test-only`, and `git diff --check` passed before submission.

## 2026-09-09 23:55 PDT

### Experiment 1: surface winds

- **Status:** Postponed by decision. No surface-wind data, preprocessing, model training, or figure update was submitted.

### Experiment 1: `to500` and revised `input4b`

- **Status:** Submitted. Added `to500` from CESM ocean `TEMP` (`degC`) at native `z_t` index 32: 48,273.671875 cm = 482.737 m, the nearest level to 500 m. `input3h_to500` uses SIC + `to500`; existing `input4b` now uses SIC + SST + `ohc200` + `to500` while retaining its established artifact and experiment names.
- **Safety:** Backed up 41 existing `exp1_input4b` checkpoint files to `/oak/stanford/groups/earlew/yuchen/sicpred/sicpred_models/exp1_input4b_pre_to500_20260909_235411_PDT` before retraining.
- **Jobs:** Download array `42701210` (14 members) → preprocessing array `42701215` (revised `input4b` overwrites its prior processed artifact) → GPU training array `42701253` (five seeds each) → postprocess array `42701254` (evaluation, diagnostics, ACC/RMSE bootstrap vs `input2`), all connected by `afterok` dependencies. Jobs are queued; no results are available yet.


## 2026-09-10 00:01 PDT

### Task 2: `icethick` min--max range triage

- **Status:** Complete as a read-only analysis in `experiments/exp1_inputs/detrending_triage/input3g_icethick_triage.ipynb`. New cells separately recompute pre-detrending min--max extrapolation from raw thickness and training-member min/max statistics, so this diagnosis is distinct from the historical detrending-leakage artifact.
- **Evidence:** 61,068 finite grid-cell/month ranges include 27,867 exact-zero ranges; the smallest positive range is `4.60e-18 m` (1st percentile `1.50e-7 m`, median `0.731 m`). Of held-out pre-detrending values, 240/10,884,840 validation values and 645/21,769,680 test values exceed `|z|=10`; the corresponding maxima are `2.69e11` and `4.84e13`. The smallest-range cells have zero training thickness-present fraction (>1 cm), consistent with marginal/ice-free contexts.
- **Candidate policies documented, not selected:** symmetric `|z|<=10` clipping and positive denominator floors of `1e-6`, `1e-4`, `1e-3`, or `1e-2 m`; the notebook reports their affected fractions and resulting maxima. Exact-zero ranges and any physical mask/floor/cap remain decisions requiring user approval.
- **Artifacts:** No production preprocessing, normalized data, model-ready pairs, models, predictions, diagnostics, figures, or Slurm jobs were changed. The new cells were executed successfully against existing artifacts.

## 2026-09-10 10:49 PDT

### Experiment 2 historical job audit

- **Status:** Preprocessing array `42700825` failed in all four tasks during Conda activation because nounset was enabled before activation (`MKL_INTERFACE_LAYER: unbound variable`). Dependent training array `42700827` and postprocessing array `42700828` were cancelled without running.
- **Artifacts:** No normalized inputs, data pairs, checkpoints, or predictions for the four reanalysis-volume configurations were created.


## 2026-09-10 11:04 PDT

### Experiment 1: `to500` prior-job audit

- **Verified outcome:** Download array `42701210` failed for all 14 tasks with exit code 1 after 10--22 seconds; each task stopped during Conda activation because `MKL_INTERFACE_LAYER` was unbound. Dependent preprocessing `42701215`, GPU training `42701253`, and postprocessing `42701254` were cancelled without starting.
- **Artifacts:** The failed chain produced no new `to500` model or prediction artifacts. The prior `input4b` checkpoints remain preserved in two timestamped archive directories; the canonical `exp1_input4b` model directory is currently absent.
- **Current status:** No replacement jobs have been submitted. Resubmission remains on hold pending explicit approval of the revised resource and dependency layout.
