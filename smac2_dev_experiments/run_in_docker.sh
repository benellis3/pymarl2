#!/bin/bash

# USAGE:
# ./smac2_dev_experiments/run_in_docker.sh <GPU> <SMAC_VERSION> <SCRIPT>

HASH=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 4 | head -n 1)
GPU=$1
SMAC_VERSION=$2
name=${USER}_pymarl_GPU_${GPU}_${HASH}

echo "Launching container named '${name}' on GPU '${GPU}'"
# Launches a docker container using our image, and runs the provided command

if hash nvidia-docker 2>/dev/null; then
  cmd=nvidia-docker
else
  cmd=docker
fi

# Fixes for wandb.
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

NV_GPU="$GPU" ${cmd} run -d \
    --name "$name" \
    --user "$(id -u)" \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -e PYTHONPATH=/home/ms21sm/pymarl2/src:/home/ms21sm/pymarl2/smac2_dev_experiments \
    -e LC_ALL=C.UTF-8 \
    -e LANG=C.UTF-8 \
    -v "$(pwd)":/home/ms21sm/pymarl2 \
    "pymarl:ms21sm_smac_v${SMAC_VERSION}" \
    ${@:3}

# E.g. commands.

# Run training:
# ./smac2_dev_experiments/run_in_docker.sh 0 1 python smac2_dev_experiments/experiments/obs_masking_effects/main.py
# ./smac2_dev_experiments/run_in_docker.sh 0 2 python smac2_dev_experiments/experiments/obs_masking_effects/main.py

# Create a wandb sweep:
# ./smac2_dev_experiments/run_in_docker.sh 0 1 wandb sweep smac2_dev_experiments/experiments/obs_masking_effects/masking_sweep.yaml
# ./smac2_dev_experiments/run_in_docker.sh 0 2 wandb sweep smac2_dev_experiments/experiments/obs_masking_effects/masking_sweep.yaml

# Run a sweep (hyperparameter optim, dist experiments, ...):
# ./smac2_dev_experiments/run_in_docker.sh 0 1 wandb agent oxwhirl/SMAC-v2-obs-masking-effects/u1pz95ld
