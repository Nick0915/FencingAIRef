#!/bin/bash
#SBATCH --partition=gpuq                    # need to set 'gpuq' or 'contrib-gpuq'  partition
#SBATCH --qos=gpu                           # need to select 'gpu' QOS or other relvant QOS
#SBATCH --job-name=label-sabre-vids-gpu
#SBATCH --output=/scratch/%u/%x-%N-%j.out   # Output file
#SBATCH --error=/scratch/%u/%x-%N-%j.err    # Error file
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5                 # number of cores needed
#SBATCH --gres=gpu:1g.10gb:1                # up to 8; only request what you need
#SBATCH --mem-per-cpu=3500M                 # memory per CORE; total memory is 1 TB (1,000,000 MB)
#SBATCH --export=ALL 
#SBATCH --time=0-06:00:00                   # set to 2hr; please choose carefully

set echo
umask 0027

# to see ID and state of GPUs assigned
nvidia-smi

cd $SCRATCH/FencingAIRef
module load gnu10                           
module load miniconda3
module load git

source ~/.bashrc
conda activate ML
pip install -r requirements.txt

time python3 2_label_vids.py
