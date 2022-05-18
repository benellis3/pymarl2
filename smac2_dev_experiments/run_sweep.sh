#!/bin/bash

# USAGE ./smac2_dev_experiments/run_sweep.sh <SWEEP_ID> <GPUS>

SWEEP_ID=$1
for gpu in "${@:2}"
do
    ./smac2_dev_experiments/run_in_docker.sh $gpu wandb agent $SWEEP_ID
done

# small datasets: oxwhirl/SMAC2-masking-runs/cndzuvih
# debug small: oxwhirl/SMAC2-masking-runs/zihy7ke7
# medium datasets: oxwhirl/SMAC2-masking-runs/vqajbssl
# large datasets: oxwhirl/SMAC2-masking-runs/g70rsnj2
