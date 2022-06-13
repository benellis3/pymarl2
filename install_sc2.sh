#!/bin/bash
# Install SC2 and add the custom maps

mkdir 3rdparty
cd 3rdparty

export SC2PATH=`pwd`'/StarCraftII'
echo 'SC2PATH is set to '$SC2PATH

if [ ! -d $SC2PATH ]; then
        echo 'StarCraftII is not installed. Installing now ...';
        wget http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip
        unzip -P iagreetotheeula SC2.4.10.zip
        rm -rf SC2.4.10.zip
else
        echo 'StarCraftII is already installed.'
fi

echo 'Adding SMAC maps.'
MAP_DIR="$SC2PATH/Maps/"
echo 'MAP_DIR is set to '$MAP_DIR

if [ ! -d $MAP_DIR ]; then
        mkdir -p $MAP_DIR
fi

if [ -d $MAP_DIR/"SMAC_Maps" ]; then
        rm -r $MAP_DIR/"SMAC_Maps"
fi

cd ..
git clone git@github.com:benellis3/smac.git
cd ./smac
git checkout smac-v2-feature-names
cd ..
mv smac/smac/env/starcraft2/maps/SMAC_Maps $MAP_DIR
rm -rf ./smac

echo 'StarCraft II and SMAC are installed.'
