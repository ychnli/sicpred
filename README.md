# Seasonal Antarctic sea ice prediction with machine learning

## Setting up the environment
TODO

## Downloading data
**CESM2 Large Ensemble**: data is downloaded from AWS cloud CESM2-LE, regridded, and saved per ensemble member in `src/download/download_cesm_data.py`. 
1. First, go to `src/config_cesm.py` and set the global variable `SCRATCH_DATA_DIR` to a file path. The regridded CESM2 data will be saved here. 
2. Next, run the download script with 
```bash
python -m src.download.download_cesm_data
```

Note that this script also supports parallel downloads via an array job, if you are running on HPC. In total, downloading and regridding CESM data takes a few hours; the data is ~6.0 GB.

**Observational data**: TODO

## Data preprocessing

## Training


# Old stuff (cleanup)
This repository is organized as follows. `src` contains python and shell scripts executable on Sherlock. `notebooks` contains drafts of models, loss functions, and some of the code used to preview training data and/or model analysis. `experiments` contains the scripts used to train models and analyze their outputs.

### General tips for running on Sherlock

All scripts and notebooks *except regridding scripts* can be run with the virtual environment `sicpred_env` on Sherlock. Currently this environment lives in `$HOME` (``/home/users/yucli``). The regridding scripts can be run with conda base environment. This is because the `xesmf` package, which is used in regridding, is not available on pip (building from source proved to be a doomed effort).

To activate the environment from `$HOME`, run `source start.sh`.

All scripts should be run from `sicpred`, *not* `src` or `experiments`. This is because the module import structure relies on the path where you call the script. 

## Scripts contained in this repo

#### Non-executable modules (generally contain only functions/classes)

`src/config.py` contains settings including train/val/test split, input data configurations, the underlying grid, and the directory for all project files. You probably don't need to change too much in here, unless you are reconfiguring input data.

`src/util.py` contains ubiquitous functions that are called in data preprocessing

`src/models_util.py` contains functions that are used for training models. It also includes a `SeaIceDataset` class, which is used by PyTorch to load input data from saved data pairs.

`src/models.py` contains PyTorch models used for training and the linear trend model 

#### Executable scripts

To preprocess the data, run the following scripts: 

`src/download_era5_data_parallel.sh` calls `download_era5_data.py` in parallel for all variables that need to be downloaded. By default if the script detects a file already present at the download destination, it will **not** overwrite the file. Currently you can only override this by changing the function call in the `.py` script to `overwrite=True`.

`src/regrid_era5_data_parallel.sh` calls `regrid_era5_data.py` in parallel for all variables that need regridding. Same thing applies for overwriting. 

`src/preprocess_data.py` preps data from the regridded outputs to input-target pairs saved as `.h5` files. By default, overwrite is off for all preprocessing functions, so the script will skip steps that are already completed. *Note: if you are running this script to generate new data pair files, 32 GB of RAM is necessary to prevent out-of-memory issues.* 

#### Experiments
`experiments/anom_pred_models_ensemble.py` trains an ensemble of 15 UNetRes3 models to predict sea ice concentration anomalies 

`experiments/anom_pred_models_hyperparam.py` sweeps across some hyperparameters for the anomaly prediction model

`experiments/anom_pred_models_diagnostics.ipynb` contains code to compute some diagnostics for the anomaly model ensemble 

`experiments/train_regression_models.ipynb` trains the original sea ice concentration prediction model across various hyperparameter settings. 

`experiments/toy_regression.ipynb` trains a simple 1 layer MLP with a mean and variance head on a toy regression dataset using NLL and $\beta$-NLL loss. 

#### Other notebooks
`notebooks/prepdata.ipynb` visualize the input data distributions 

`notebooks/evaluate_models.ipynb` older code to visualize outputs of the original model outputs (non-anomaly prediction). Also plot the results of an initial hyperparameter sweep 

`notebooks/illustrations.ipynb` visualize anomaly persistence (a figure used in the end-of-summer presentation)

`notebooks/models_testing.ipynb` contains draft code for the uncertainty quantification model. Currently, not set up to save model weights. TODO: write this into a script in the experiments folder 