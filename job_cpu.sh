#!/bin/sh
#BSUB -J Whipping
#BSUB -q hpc
#BSUB -n 32
#BSUB -W 12:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=10GB]"
#BSUB -u baixianger@gmail.com
#BSUB -B
#BSUB -N
#BSUB -o RL_%J.out
#BSUB -e RL_%J.err
# -- end of LSF options --

# export CUDA_VISIBLE_DEVICES=1,3
# nvidia-smi
# Load the cuda module
# module load cuda/11.8
# cd /work3/s213120/whipping_targets

