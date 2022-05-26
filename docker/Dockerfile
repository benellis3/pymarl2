FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

RUN apt-get -y update
RUN apt-get -y install git
ADD install_dependencies.sh /source/
RUN bash -c /source/install_dependencies.sh

RUN pip install pymongo

ENV smac_ver 1
ENV SC2PATH /home/ms21sm/pymarl2/3rdparty/StarCraftII

ARG UID
RUN useradd -u $UID --create-home ms21sm
USER ms21sm
WORKDIR /home/ms21sm/pymarl2
