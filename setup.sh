#!/usr/bin/env bash
sudo apt-get update;
sudo apt-get upgrade -y;

sudo apt-get -y install python3-dev python3-pip git libhdf5-dev python3-tk libfuzzy-dev libffi-dev graphviz lzma

sudo pip3 install numpy scipy sklearn keras theano tensorflow h5py matplotlib gevent requests ssdeep

git clone https://github.com/jras/IntroductionToMachineLearningForSecurityPros.git

cd IntroductionToMachineLearningForSecurityPros

find . -type f -name \*.lzma -exec lzma -d {} \;
