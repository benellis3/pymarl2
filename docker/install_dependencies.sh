#!/bin/bash

echo "Installing SMAC version ${1}"

pip install --upgrade pip
pip install --ignore-installed six
pip install sacred numpy scipy gym==0.10.8 matplotlib seaborn pyyaml pygame pytest probscale imageio snakeviz tensorboard-logger wandb tqdm pymongo

if [ "${1}" = "1" ]
then
  # SMAC (v1).
  pip install git+https://github.com/skandermoalla/smac.git@feature-names
else
  # SMACv2
  pip install git+https://github.com/benellis3/smac.git@smac-v2-feature-names
fi
