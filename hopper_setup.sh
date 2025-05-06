#!/bin/bash

cd $SCRATCH/FencingAIRef
module load gnu10 git miniconda3 ffmpeg
source ~/.bashrc
conda activate ML
pip install -r requirements.txt

salloc --partition=normal --time=0-08:00:00 --nodes=1 --cpus-per-task=64 --mem-per-cpu=2GB
