###################################################################################
# This script runs all scripts associated with the variable importance experiment
# (exp1) except for model training.
###################################################################################

#!/bin/bash
set -e

bash experiments/exp1_inputs/preprocess.sh
bash experiments/exp1_inputs/evaluate.sh
bash experiments/exp1_inputs/diagnostics.sh
bash experiments/exp1_inputs/permute_and_predict.sh