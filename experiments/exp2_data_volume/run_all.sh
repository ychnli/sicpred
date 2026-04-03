###################################################################################
# This script runs all scripts associated with the data scaling experiment
# (exp2). A GPU is recommended for training the models.
###################################################################################

#!/bin/bash
set -e

bash experiments/exp2_data_volume/preprocess.sh
bash experiments/exp2_data_volume/train.sh
bash experiments/exp2_data_volume/diagnostics.sh