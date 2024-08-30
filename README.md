## Code for seasonal Antarctic sea ice prediction with machine learning

This repository is organized as follows. `src` contains python and shell scripts executable on Sherlock. `notebooks` contains drafts of models, loss functions, and some of the code used to preview training data and/or model analysis. `experiments` contains the scripts used to train models and analyze their outputs.

### General tips for running on Sherlock

All scripts and notebooks *except regridding scripts* can be run with the virtual environment `sicpred_env` on Sherlock. Currently this environment lives in `$HOME` (``/home/users/yucli``). The regridding scripts can be run with conda base environment. This is because the `xesmf` package, which is used in regridding, is not available on pip (building from source proved to be a doomed effort). 

To activate the environment from `$HOME`, run `source start.sh`. 

### Scripts contained in this repo

`src/config.py` contains settings including train/val/test split, input data configurations, the underlying grid, and the directory for all project files. You probably don't need to change too much in here, unless you are reconfiguring input data.
