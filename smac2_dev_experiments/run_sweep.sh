#!/bin/bash

# USAGE ./smac2_dev_experiments/run_sweep.sh <SMAC_VERSION> <SWEEP_ID> <GPUS>

SMAC_VERSION=$1
SWEEP_ID=$2
for gpu in "${@:3}"
do
    ./smac2_dev_experiments/run_in_docker.sh $gpu $SMAC_VERSION wandb agent $SWEEP_ID &
done

# small datasets: oxwhirl/SMAC2-masking-runs/cndzuvih
# debug small: oxwhirl/SMAC2-masking-runs/zihy7ke7
# medium datasets: oxwhirl/SMAC2-masking-runs/vqajbssl
# large datasets: oxwhirl/SMAC2-masking-runs/g70rsnj2
