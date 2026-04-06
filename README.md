# Seasonal Antarctic sea ice prediction with machine learning

## 1) Setting up the environment
First, clone this git repository. The environment dependency file is `environment.yml`. To recreate the environment using the conda package manager, do
```bash
conda env create -f environment.yml
```
Activate the environment by doing
```bash
conda activate sicpred_env_conda
```

## 2) Downloading data
**CESM2 Large Ensemble**: CESM2-LE data is downloaded from AWS cloud, regridded, and saved per ensemble member in `src/download/download_cesm_data.py`. 
1. First, go to `src/config_cesm.py` and set the global variable `DATA_DIRECTORY` to the desired file path. The regridded CESM2 data will be saved there. 
2. Next, run the download script with 
```bash
python -m src.download.download_cesm_data
```

Note that this script also supports parallel downloads via an array job if you are running on HPC. In total, downloading and regridding CESM data can take up to a few hours; the final data is ~6.0 GB.

**Observational data**: the regridded ERA5 sea ice concentration used in the finetuning experiment can be found in the Zenodo. This should be downloaded and placed in a directory called `obs_data` in your `DATA_DIRECTORY`.

**Model weights**: if you wish to recreate the results of the paper *without having to retrain the models*, the model weights can be downloaded from Zenodo. Once they are downloaded, go to `src/config_cesm.py` and set the global variable `MODEL_DIRECTORY` to the folder containing model weights.

## 3) Configuring the repository
Before running any experiments, you need to create and set the paths used for saving model checkpoints and results in `src/config_cesm.py`:
- `PROCESSED_DATA_DIRECTORY` will store model-ready standardized data and concatenated input-output data pairs
- `PREDICTIONS_DIRECTORY` will store model predictions
- `ANALYSIS_RESULTS_DIRECTORY` will store model diagnostics

## 4) Running experiments
There are three experiments, each of which can be run via a shell script:
- Experiment 1 (`exp1_inputs`): variable importance experiment (Section 3.1, 4.1)
- Experiment 2 (`exp2_data_volume`): training data scaling experiment (Section 3.2)
- Experiment 3 (`exp3_obs`): finetuning on observations (Section 3.3)

**Recommended resources**: In general, a GPU is recommended for training and evaluating models (especially training). 64 GB of RAM is recommended.

To generate all results *including model training*, run (from the directory root):
```bash
bash experiments/exp1_inputs/run_all.sh
bash experiments/exp2_data_volume/run_all.sh
bash experiments/exp3_obs/run_all.sh
```
Note that exp2 needs to be run before exp3 (since we use one of the models trained in exp2 as a pretrained checkpoint in exp3), but this is the only interdependency.

To generate all results *excluding model training* (i.e., you have downloaded the model weights and set `MODEL_DIRECTORY` accordingly), run (from the directory root):
```bash
bash experiments/exp1_inputs/run_all_notrain.sh
bash experiments/exp2_data_volume/run_all_notrain.sh
bash experiments/exp3_obs/run_all_notrain.sh
```
Note that the scripts are configured by default to find already-saved results, so if the script is interrupted, rerunning it will resume where it last left off.

## 5) Creating figures
The figures can be reproduced by running the notebooks `exp1_figures.ipynb`, `exp2_figures.ipynb`, `exp3_figures.ipynb`, and `supp_figures.ipynb`. 