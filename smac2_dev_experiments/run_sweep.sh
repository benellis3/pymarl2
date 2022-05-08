#!/bin/bash

# USAGE ./smac2_dev_experiments/run_sweep.sh <SWEEP_ID>

SWEEP_ID=$1
./smac2_dev_experiments/run_in_docker.sh 0 wandb agent $SWEEP_ID
./smac2_dev_experiments/run_in_docker.sh 1 wandb agent $SWEEP_ID
./smac2_dev_experiments/run_in_docker.sh 2 wandb agent $SWEEP_ID
./smac2_dev_experiments/run_in_docker.sh 3 wandb agent $SWEEP_ID
./smac2_dev_experiments/run_in_docker.sh 4 wandb agent $SWEEP_ID
./smac2_dev_experiments/run_in_docker.sh 5 wandb agent $SWEEP_ID
# Only 6 GPUS as CPU load already exceed 100% with that.
