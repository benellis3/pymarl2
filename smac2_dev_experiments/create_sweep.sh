#!/bin/bash

# USAGE: ./smac2_dev_experiments/create_sweep.sh PATH_TO_SWEEP_NAME
# eg ./smac2_dev_experiments/create_sweep.sh obs_masking_effects/sweeps/smac_2/example.yaml

./smac2_dev_experiments/run_in_docker.sh 0 1 wandb sweep smac2_dev_experiments/experiments/$1
# Then get the sweep id in docker logs.
sleep 3
docker logs $(docker ps -a | sed -n 2p | grep -oE '[^ ]+$')