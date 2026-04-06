###################################################################################
# This script runs all scripts associated with the data scaling experiment
# (exp2) except for training the model.
###################################################################################

#!/bin/bash
set -e

bash experiments/exp2_data_volume/preprocess.sh
bash experiments/exp2_data_volume/evaluate.sh
bash experiments/exp2_data_volume/diagnostics.sh