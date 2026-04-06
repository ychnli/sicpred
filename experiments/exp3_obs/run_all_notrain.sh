###################################################################################
# This script runs all scripts associated with the finetuning experiment (exp3)
# except for training the models.
###################################################################################

#!/bin/bash
set -e

bash experiments/exp3_obs/preprocess.sh
bash experiments/exp3_obs/evaluate.sh
bash experiments/exp3_obs/diagnostics.sh