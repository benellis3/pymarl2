FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
MAINTAINER Christian Schroeder de Witt

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update && apt-get install -y git

RUN pip install --upgrade pip
RUN pip install numpy scipy pyyaml matplotlib imageio pygame tensorboard-logger ruamel.base ryd wandb

RUN mkdir /install
WORKDIR /install

RUN pip install jsonpickle==0.9.6
## install Sacred (from OxWhirl fork)
#RUN pip install setuptools
#RUN git clone https://github.com/oxwhirl/sacred.git /install/sacred && cd /install/sacred && python setup.py install

# Install pymongo
RUN pip install pymongo

## -- SMAC
ENV smac_ver 1
RUN pip install "protobuf<3.21" git+https://github.com/oxwhirl/smacv2.git@posmac
ENV SC2PATH /home/jonook/src/pymarl2/3rdparty/StarCraftII

RUN git clone https://github.com/koulanurag/ma-gym.git /install/ma-gym && cd /install/ma-gym && pip install -e .

ARG UID
RUN useradd -u $UID --create-home jonook
USER jonook
WORKDIR /home/jonook/src/pymarl2

RUN echo 'set editing-mode vi' >> /home/jonook/.inputrc
RUN echo 'set keymap vi' >> /home/jonook/.inputrc