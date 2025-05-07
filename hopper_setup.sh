#!/bin/bash

cd $SCRATCH/FencingAIRef
module load gnu10 git miniconda3 ffmpeg
source ~/.bashrc
conda activate ML
pip install -r requirements.txt

salloc --partition=normal --time=0-08:00:00 --nodes=1 --cpus-per-task=64 --mem-per-cpu=2GB

salloc -p gpuq -q gpu --nodes=1 --ntasks-per-node=12 --gres=gpu:1g.10gb:1 --mem=15gb -t 0-02:00:00

