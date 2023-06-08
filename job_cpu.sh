#!/bin/sh
#BSUB -J Whipping
#BSUB -q hpc
#BSUB -n 32
#BSUB -W 05:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=10GB]"
###BSUB -R "select[model == XeonGold6342]"
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

# ppo算法在不同的奖励函数在单步任务模式下的表现实验
python3 train.py wandb_group=ppo_single task=SingleStepTaskSimple algo=ppo exp_name=reward_design task.target=0 task.reward_type=0 'algo.hidden_dims=[32, 16]' algo.num_envs=64
# python3 train.py wandb_group=ppo_single task=SingleStepTaskSimple algo=ppo exp_name=reward_desing task.target=0 task.reward_type=1 'algo.hidden_dims=[32, 16]' algo.num_envs=128
# python3 train.py wandb_group=ppo_single task=SingleStepTaskSimple algo=ppo exp_name=reward_design task.target=0 task.reward_type=2 'algo.hidden_dims=[32, 16]' algo.num_envs=128
# python3 train.py wandb_group=ppo_single task=SingleStepTaskSimple algo=ppo exp_name=reward_design task.target=0 task.reward_type=3 'algo.hidden_dims=[32, 16]' algo.num_envs=128
# python3 train.py wandb_group=ppo_single task=SingleStepTaskSimple algo=ppo exp_name=reward_design task.target=0 task.reward_type=4 'algo.hidden_dims=[32, 16]' algo.num_envs=64