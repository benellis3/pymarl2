FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

RUN apt-get -y update
RUN apt-get -y install git vim

RUN pip install --upgrade pip
RUN pip install --ignore-installed six
RUN pip install sacred numpy scipy gym==0.10.8 matplotlib seaborn pyyaml pygame pytest probscale imageio snakeviz tensorboard-logger wandb
RUN pip install pymongo

ENV smac_ver 1
RUN pip install "protobuf<3.21" -U git+https://github.com/oxwhirl/smacv2.git@posmac
ENV SC2PATH /home/jonook/src/pymarl2/3rdparty/StarCraftII

ARG UID
RUN useradd -u $UID --create-home jonook
USER jonook
WORKDIR /home/jonook/src/pymarl2

RUN echo 'set editing-mode vi' >> /home/jonook/.inputrc
RUN echo 'set keymap vi' >> /home/jonook/.inputrc