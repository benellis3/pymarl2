#!/bin/bash
# Install PyTorch and Python Packages

# conda create -n pymarl python=3.7 -y
# conda activate pymarl

pip install --upgrade pip
pip install --ignore-installed six
pip install sacred numpy scipy gym==0.10.8 matplotlib seaborn pyyaml pygame pytest probscale imageio snakeviz tensorboard-logger wandb tqdm
# SMAC v1
# pip install git+https://github.com/skandermoalla/smac.git@feature-names
# SMAC v2
pip install git+https://github.com/benellis3/smac.git@smac-v2-feature-names
