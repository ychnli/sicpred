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