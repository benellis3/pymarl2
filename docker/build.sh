#!/bin/bash

if [ ! $1 ]; then
    echo "Please enter the version of smac."
    echo "e.g. './build.sh 1' or './build.sh 2"
    exit 1
fi

echo "Building Dockerfile with image name pymarl for smac version ${1}"
docker build --no-cache --build-arg UID=$UID --build-arg SMAC_VERSION=$1 -t "pymarl:ms21sm_smac_v${1}" .
