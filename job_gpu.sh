#!/bin/sh
#BSUB -J Whipping
#BSUB -q gpuv100
#BSUB -n 24
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -u baixianger@gmail.com
#BSUB -B
#BSUB -N
#BSUB -o RL_%J.out
#BSUB -e RL_%J.err

# this only for v100 with a 32Gb memory
# -- end of LSF options --

# export CUDA_VISIBLE_DEVICES=1,3
# nvidia-smi
# Load the cuda module
# module load cuda/11.8
# cd /work3/s213120/whipping_targets

############# 单步任务模式下击打随机目标的表现实验 需要很长时间 24小时
python3 train.py wandb_group=random_hit_single task=SingleStepTaskSimple algo=ppo exp_name=random_hit task.target=1 task.reward_type=5 'algo.hidden_dims=[64, 32, 16]' algo.num_envs=128 algo.num_updates=1000