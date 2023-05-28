"""Test the trained agent in dm_control environment"""
from dm_control import composer, viewer
import numpy as np
import torch
from env.task import SingleStepTask
from RL.ppo_continuous_action import Agent

single_step_task = SingleStepTask(ctrl_type='position',
                                  whip_type=0,
                                  target=0)
single_step_task.set_timesteps(0.04, 0.02)
env = composer.Environment(single_step_task)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")
agent = torch.load("SingleStepTask-v0__HPCtest__42__1685188963-update400.pth",
                   map_location=device)
# The operator 'aten::_sample_dirichlet' is not currently implemented for the MPS device. 
# If you want this op to be added in priority during the prototype phase of this feature, 
# please comment on https://github.com/pytorch/pytorch/issues/77764. 
# As a temporary fix, you can set the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as a fallback for this op. 
# WARNING: this will be slower than running natively on MPS.


def my_policy(time_step):
    """wrap agent.policy to fit dm_control"""
    init_obs = np.hstack(list(time_step.observation.values()))
    init_obs = torch.Tensor(init_obs).to(device)
    action, _, _ = agent.policy(init_obs)
    action = action[0].cpu().numpy()
    return action


viewer.launch(env, policy=my_policy)
