#!/bin/sh
#BSUB -q gpuv100
#BSUB -J Whipping
#BSUB -n 16
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 5:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -u baixianger@gmail.com
#BSUB -B
#BSUB -N
#BSUB -o ppo_%J.out
#BSUB -e ppo_%J.err
# -- end of LSF options --

## export CUDA_VISIBLE_DEVICES=1,3
nvidia-smi
## Load the cuda module
# module load cuda/11.8

# python3 ./utils/data_loader.py
python3 train.py

## Tutorial about LSF
# submit job
# bsub < jobscript.sh
# check job status
# bjobs
# check job output
# bpeek <job_id>
# cancel job
# bkill <job_id>

## template for batch jobs
# ref: https://www.hpc.dtu.dk/?page_id=2759
## python package FAQ
# ref: https://www.hpc.dtu.dk/?page_id=3678
## LSF job management
# ref: https://www.hpc.dtu.dk/?page_id=1519
