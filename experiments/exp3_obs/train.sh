###################################################################################
# This script trains models on the observational dataset for the finetuning experiment.
#
# Results are saved to MODEL_DIRECTORY which should be set in config_cesm.py
# 
# A GPU is recommended for this script
###################################################################################

# This trains the baseline model (only on obs)
python3 -m src.models.train --config exp3_obs:obs_input2 --members 20

# first, get the location of the model checkpoints by extracting it from config_cesm.py
MODEL_DIRECTORY=$(python3 - <<EOF
from src.config_cesm import MODEL_DIRECTORY
print(MODEL_DIRECTORY)
EOF
)
PRETRAINED_PATH="$MODEL_DIRECTORY/exp2_vol4/UNetRes3_exp2_vol4_member_0_best.pth"
echo "Using $PRETRAINED_PATH as pretrained checkpoint"

# This finetunes the pretrained model
python3 -m src.models.train --config exp3_obs:obs_input2_finetune --pretrained "$PRETRAINED_PATH" --members 20
