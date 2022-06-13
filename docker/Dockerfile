FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

RUN apt-get -y update
RUN apt-get -y install git
ARG SMAC_VERSION
ADD install_dependencies.sh /source/
RUN bash -c "/source/install_dependencies.sh ${SMAC_VERSION}"

ARG USER
ARG UID
RUN useradd -u $UID --create-home $USER
USER $USER

ENV SC2PATH /home/$USER/pymarl2/3rdparty/StarCraftII
WORKDIR /home/$USER/pymarl2
