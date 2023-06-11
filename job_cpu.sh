#!/bin/sh
#BSUB -J Whipping
#BSUB -q hpc
#BSUB -n 32
#BSUB -W 10:00
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

############ ppo算法在不同的奖励函数在单步任务模式下的表现实验 五种奖励设计 高度和速度
# python3 train.py wandb_group=ppo_single task=SingleStepTaskSimple algo=ppo exp_name=reward_design task.target=0 task.reward_type=0 'algo.hidden_dims=[32, 16]' algo.num_envs=64
# python3 train.py wandb_group=ppo_single task=SingleStepTaskSimple algo=ppo exp_name=reward_desing task.target=0 task.reward_type=1 'algo.hidden_dims=[32, 16]' algo.num_envs=64
# python3 train.py wandb_group=ppo_single task=SingleStepTaskSimple algo=ppo exp_name=reward_design task.target=0 task.reward_type=2 'algo.hidden_dims=[32, 16]' algo.num_envs=64
# python3 train.py wandb_group=ppo_single task=SingleStepTaskSimple algo=ppo exp_name=reward_design task.target=0 task.reward_type=3 'algo.hidden_dims=[32, 16]' algo.num_envs=64
# 需要12小时 300个epoch
# python3 train.py wandb_group=ppo_single task=SingleStepTaskSimple algo=ppo exp_name=reward_design task.target=0 task.reward_type=4 'algo.hidden_dims=[32, 16]' algo.num_envs=64 algo.num_updates=300
# python3 train.py wandb_group=ppo_single task=SingleStepTaskSimple algo=ppo exp_name=reward_design task.target=0 task.reward_type=5 'algo.hidden_dims=[32, 16]' algo.num_envs=64 algo.num_updates=300

############# ppo算法在多步任务模式下的奖励表现实验 四种奖励
# python3 train.py wandb_group=ppo_multi task=MultiStepTaskSimple algo=ppo exp_name=reward_design_%J task.target=0 task.reward_type=0 algo.num_envs=64 algo.save_freq=500 algo.num_updates=3000
# python3 train.py wandb_group=ppo_multi task=MultiStepTaskSimple algo=ppo exp_name=reward_design_%J task.target=0 task.reward_type=1 algo.num_envs=64 algo.save_freq=500 algo.num_updates=3000
# python3 train.py wandb_group=ppo_multi task=MultiStepTaskSimple algo=ppo exp_name=reward_design_%J task.target=0 task.reward_type=2 algo.num_envs=64 algo.save_freq=500 algo.num_updates=3000
# python3 train.py wandb_group=ppo_multi task=MultiStepTaskSimple algo=ppo exp_name=reward_design_%J task.target=0 task.reward_type=3 algo.num_envs=64 algo.save_freq=500 algo.num_updates=3000

############# 各种算法在多步任务模式下采用reward0规则的表现实验
# python3 train.py wandb_group=multi_diff_algo task=MultiStepTaskSimple algo=ppo  exp_name=diff_ppo  task.target=0 task.reward_type=1 algo.num_envs=64 algo.save_freq=500  algo.num_updates=2000
# python3 train.py wandb_group=multi_diff_algo task=MultiStepTaskSimple algo=ddpg exp_name=diff_ddpg task.target=0 task.reward_type=1 algo.num_envs=64 algo.save_freq=1000 algo.num_updates=50000 algo.save_freq=10000
# python3 train.py wandb_group=multi_diff_algo task=MultiStepTaskSimple algo=td3  exp_name=diff_td3  task.target=0 task.reward_type=1 algo.num_envs=64 algo.save_freq=1000 algo.num_updates=50000 algo.save_freq=10000
# python3 train.py wandb_group=multi_diff_algo task=MultiStepTaskSimple algo=sac  exp_name=diff_sac  task.target=0 task.reward_type=1 algo.num_envs=64 algo.save_freq=1000 algo.num_updates=50000 algo.save_freq=10000

############# 单步任务模式下击打随机目标的表现实验 需要很长时间 24小时
# python3 train.py wandb_group=random_hit_single task=SingleStepTaskSimple algo=ppo exp_name=random_hit_%J task.target=1 task.reward_type=5 'algo.hidden_dims=[64, 32, 16]' algo.num_envs=128 algo.num_updates=600


# 一个比较dirty的方法就是这这里启动jupyterlab，然后在jupyterlab里跑你的任务
# jupyter lab --port=44000 --ip=$HOSTNAME --no-browser