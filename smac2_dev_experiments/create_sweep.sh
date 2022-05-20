#!/bin/bash

# USAGE: ./smac2_dev_experiments/create_sweep.sh SWEEP_NAME

./smac2_dev_experiments/run_in_docker.sh 0 wandb sweep smac2_dev_experiments/experiments/$1
# Then get the sweep id in docker logs.
sleep 3
docker logs $(docker ps -a | sed -n 2p | grep -oE '[^ ]+$')