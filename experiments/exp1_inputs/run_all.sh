###################################################################################
# This script runs all scripts associated with the variable importance experiment
# (exp1). A GPU is recommended for training the models.
###################################################################################

#!/bin/bash
set -e

bash experiments/exp1_inputs/preprocess.sh
bash experiments/exp1_inputs/train.sh
bash experiments/exp1_inputs/diagnostics.sh
bash experiments/exp1_inputs/permute_and_predict.sh