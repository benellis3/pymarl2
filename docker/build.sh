#!/bin/bash

echo 'Building Dockerfile with image name pymarl'
docker build --no-cache -t pymarl:ben_smac .
