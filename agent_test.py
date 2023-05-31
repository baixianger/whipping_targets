"""Test the trained agent in dm_control environment"""
from dm_control import composer, viewer
import numpy as np
import torch
from env.easy_task import SingStepTaskSimple

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
agent = torch.load("checkpoints/SingStepTaskSimple__HPCtest__42__1685454905-update30.pth", map_location=device)

task = SingStepTaskSimple()
task.time_limit = 1
task.set_timesteps(0.01, 0.01)
env = composer.Environment(task)
# The operator 'aten::_sample_dirichlet' is not currently implemented for the MPS device. 
# If you want this op to be added in priority during the prototype phase of this feature, 
# please comment on https://github.com/pytorch/pytorch/issues/77764. 
# As a temporary fix, you can set the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as a fallback for this op. 
# WARNING: this will be slower than running natively on MPS.

def get_action(time_step):
    """Get action from the first time step of the environment"""
    with torch.no_grad():
        obs = np.hstack([value.flatten() for _, value in time_step.observation.items()]) # flatten the observation
        obs = torch.Tensor(obs).view(1, -1).to(device)
        action, _, _ = agent.policy(obs)
        action = action[0].cpu().numpy()
    return action

time_step = env.reset()
action = get_action(time_step)

viewer.launch(env, policy=lambda time_step: action)
