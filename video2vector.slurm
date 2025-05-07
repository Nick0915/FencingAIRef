#!/bin/bash

#SBATCH --partition=gpuq                    # need to set 'gpuq' or 'contrib-gpuq'  partition
#SBATCH --qos=gpu                           # need to select 'gpu' QOS or other relvant QOS
#SBATCH --job-name=video2vector
#SBATCH --output=/scratch/%u/%x-%N-%j.out   # Output file
#SBATCH --error=/scratch/%u/%x-%N-%j.err    # Error file
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                 # number of cores needed
#SBATCH --gres=gpu:A100.40gb:1              # up to 8; only request what you need
#SBATCH --mem=16gb 	                    # memory per CORE; total memory is 1 TB (1,000,000 MB)
#SBATCH --export=ALL 
#SBATCH --time=0-08:00:00                   # set to 8hr; please choose carefully

set echo
umask 0027

# to see ID and state of GPUs assigned
# nvidia-smi

cd $SCRATCH/FencingAIRef
module load gnu10 miniconda3
source ~/.bashrc
conda activate ML
pip install -r requirements.txt

time python3 ./6_videos_to_features.py
