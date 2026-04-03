# Seasonal Antarctic sea ice prediction with machine learning

## 1) Setting up the environment
First, clone this git repository. The environment dependency file is `environment.yml`. To recreate the environment using the conda package manager, do
```bash
conda env create -f environment.yml
```

## 2) Downloading data
**CESM2 Large Ensemble**: CESM2-LE data is downloaded from AWS cloud, regridded, and saved per ensemble member in `src/download/download_cesm_data.py`. 
1. First, go to `src/config_cesm.py` and set the global variable `RAW_DATA_DIRECTORY` to the desired file path. The regridded CESM2 data will be saved there. 
2. Next, run the download script with 
```bash
python -m src.download.download_cesm_data
```

Note that this script also supports parallel downloads via an array job if you are running on HPC. In total, downloading and regridding CESM data can take up to a few hours; the final data is ~6.0 GB.

**Observational data**: the regridded ERA5 sea ice concentration used in the finetuning experiment can be found in the Zenodo [link]. 

## 3) Configuring the repository

## 4) Running the experiments

## 5) Reproducing figures
