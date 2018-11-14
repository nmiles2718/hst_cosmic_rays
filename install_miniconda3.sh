#!/usr/bin/env bash
# Run this once logged into the EC2 instance to install miniconda
curl -OL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh

bash ./Miniconda3-latest-Linux-x86_64.sh

source activate miniconda3

conda env create -aws_env ./CONFIG/aws_env.yml