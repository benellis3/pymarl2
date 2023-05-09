#!/bin/bash
# Install PyTorch and Python Packages

# conda create -n pymarl python=3.7 -y
# conda activate pymarl

pip install --upgrade pip
pip install --ignore-installed six
pip install sacred numpy scipy gym==0.10.8 matplotlib seaborn pyyaml pygame pytest probscale imageio snakeviz tensorboard-logger wandb
pip install git+https://github.com/oxwhirl/smac.git
