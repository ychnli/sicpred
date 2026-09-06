###################################################################################
# This script runs the permute-and-predict experiment (Figure 6)
###################################################################################

set -euo pipefail

vars=(sst psl z500 t2m)
lags=(lag1 lag2 lag3 lag4 lag5 lag6)

for lag in "${lags[@]}"; do
    for var in "${vars[@]}"; do
        python -m src.models.permute_and_predict --config exp1_inputs:input4 --var_name "${var}_${lag}"
        python3 -m src.models.diagnostics --config exp1_inputs:input4 --permute-var "${var}_${lag}"
    done
done

vars=(icefrac)
lags=(lag1 lag2 lag3 lag4 lag5 lag6 lag7 lag8 lag9 lag10 lag11 lag12)

for lag in "${lags[@]}"; do
    for var in "${vars[@]}"; do
        python -m src.models.permute_and_predict --config exp1_inputs:input4 --var_name "${var}_${lag}"
        python3 -m src.models.diagnostics --config exp1_inputs:input4 --permute-var "${var}_${lag}"
    done
done