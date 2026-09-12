# Experiment 2: data-volume configurations

The original vol1-vol4 configurations vary the number of training ensemble members and retain their historical ensemble-member splits. Do not change or rerun them when operating the reanalysis-volume analogue.

## Reanalysis-volume CESM analogue

These configurations isolate one CESM2-LE realization per data artifact and use the same input/target recipe as Experiment 1 input2. Their contiguous temporal split has the requested 33 training years (1968-2000), 6 validation years (2001-2006), and 7 test years (2007-2013).

| Config selector suffix | CESM member | Data/experiment suffix |
| --- | --- | --- |
| reanalysis_volume_r2i1251p1f1 | r2i1251p1f1 | reanalysis_volume_r2i1251p1f1 |
| reanalysis_volume_r2i1281p1f1 | r2i1281p1f1 | reanalysis_volume_r2i1281p1f1 |
| reanalysis_volume_r2i1301p1f1 | r2i1301p1f1 | reanalysis_volume_r2i1301p1f1 |
| reanalysis_volume_r3i1041p1f1 | r3i1041p1f1 | reanalysis_volume_r3i1041p1f1 |

Each complete selector is exp2_data_volume:<suffix>. Data names begin with seaice_plus_auxiliary_; experiment/output names begin with exp2_. All four are distinct from the original vol1-vol4 artifacts.

Training uses the current non-finetuned Experiment 3 settings: AdamW at learning rate 1e-3, weight decay 5e-2, batch size 32, at most 50 epochs, checkpoint interval 10, patience 10, and cosine scheduling with t_max=50 and eta_min=5e-5. Preprocessing uses the training-date-only detrending implementation.

Run `experiments/exp2_data_volume/submit_reanalysis_volume.sh` from the repository root to submit one four-task CPU preprocessing array, one GPU job that sequentially trains all five neural seeds for all four configurations and then evaluates the four ensembles, and one four-task CPU diagnostics array with `afterok` dependencies.
