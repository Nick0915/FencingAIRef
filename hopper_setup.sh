#!/bin/bash

cd $SCRATCH/FencingAIRef
module load gnu10 git miniconda3
source ~/.bashrc
conda activate ML
pip install -r requirements.txt
