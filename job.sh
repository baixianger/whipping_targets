#!/bin/sh
#BSUB -q gpuv100
#BSUB -J Whipping
#BSUB -n 32
#BSUB -R "span[hosts=1]"
## #BSUB -R "select[model==XeonGold6126]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 10:00
#BSUB -R "rusage[mem=4GB]"
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


python3 train.py task=TwoStepTask task.fixed_time=False
# python3 train.py
## Tutorial about LSF
# submit job
# bsub < job.sh
# check job status
# bjobs
# check job output
# bpeek <job_id>
# cancel job
# bkill <job_id>
# nodestat -F hpc/gpua100/gpuv100/gpua10
# showstart <job_id>

## template for batch jobs
# ref: https://www.hpc.dtu.dk/?page_id=2759
## python package FAQ
# ref: https://www.hpc.dtu.dk/?page_id=3678
## LSF job management
# ref: https://www.hpc.dtu.dk/?page_id=1519
