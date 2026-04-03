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

**Observational data**: the sea ice concentration used in the finetuning experiment can be found in the Zenodo [link]

