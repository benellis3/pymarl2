#!/bin/bash
set -x
HASH=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 4 | head -n 1)
#WANDB_API_KEY=$(cat $WANDB_API_KEY_FILE)
GPU=$1
SMAC_VERSION=$2
name=${USER}_pymarl_GPU_${GPU}_${HASH}

echo "Launching container named '${name}' on GPU '${GPU}' for smac version '${SMAC_VERSION}'"
# Launches a docker container using our image, and runs the provided command

if hash nvidia-docker 2>/dev/null; then
  cmd=nvidia-docker
else
  cmd=docker
fi

NV_GPU="$GPU" ${cmd} run \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    --name $name \
    --user $(id -u) \
     -v "$(pwd)":/home/ms21sm/pymarl2 \
    pymarl:ms21sm_smac_v${SMAC_VERSION} \
    ${@:3}

# Usage ./run_docker <gpu> <smac_version> <docker-run-flag> <command>