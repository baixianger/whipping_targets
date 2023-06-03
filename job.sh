#!/bin/sh
#BSUB -q gpua40
#BSUB -J Whipping
#BSUB -n 32
#BSUB -R "span[hosts=1]"
## #BSUB -R "select[model==XeonGold6126]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 5:00
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

# python3 train.py task.target=100
# python3 train.py task.target=100 task.ctrl_type=torque
# python3 train.py task=TwoStepTask task.target=100 
# python3 train.py task=MultiStepTask
python3 train.py task=SingleStepTaskSimple



## Tutorial for LSF
# submit job
# bsub < job.sh
# check job status
# bjobs
# check job output
# bpeek <job_id>
# cancel job
# bkill <job_id>
# nodestat -F hpc
# nodestat -F -g gpua100
# nodestat -F -g gpuv100
# nodestat -F -g gpua40
# nodestat -F -g gpua10
# showstart <job_id>



# We have right now 38 nodes with GPUs in our generally available LSF10-setup.
# The walltime is limited to 24 hours per job at the moment.

# 4 nodes with 2 x Tesla A100 PCIE 40 GB (owned by DTU Compute) – queuename: gpua100
# 6 nodes with 2 x Tesla A100 PCIE 80 GB (owned by DTU Compute) – queuename: gpua100
# 6 nodes with 2 x Tesla V100 16 GB (owned by DTU Compute&DTU Elektro) – queuename: gpuv100
# 8 nodes with 2 x Tesla V100 32 GB (owned by DTU Compute&DTU Environment&DTU MEK) – queuename gpuv100
# 1 nodes with 2 x Tesla A10 PCIE 24 GB (owned by DTU Compute) – queuename gpua10
# 1 nodes with 2 x Tesla A40 48 GB with NVlink (owned by DTU Compute) – queuename gpua40
# 3 nodes with 4 x Tesla V100 32 GB with NVlink (owned by DTU Compute) – queuename gpuv100
# 2 nodes with 4 x TitanX (Pascal) – queuename: gputitanxpascal (retired)
# 1 node with 4 x Tesla K80 – queuename: gpuk80 (retired)
# 1 node with 2 x Tesla K40 – queuename: gpuk40 

# 1 node with 2 x AMD Radeon Instinct MI50 16 GB gpus – not on queue
# 1 node with 2 x AMD Radeon Instinct MI25 16 GB gpus – queuename gpuamd

# For being able to run code on the Nvidia A100 please make sure to compile your code with
# cuda 11.0 or newer.

# 1 interactive V100-node reachable via voltash
# 1 interactive V100-node with NVlink reachable via sxm2sh
# 1 interactive A100-node with NVlink reachable via a100sh.


## template for batch jobs
# ref: https://www.hpc.dtu.dk/?page_id=2759
## python package FAQ
# ref: https://www.hpc.dtu.dk/?page_id=3678
## LSF job management
# ref: https://www.hpc.dtu.dk/?page_id=1519
