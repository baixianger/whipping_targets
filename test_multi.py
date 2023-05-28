"""Module test"""
import os
import numpy as np
import torch
from torchinfo import summary
from env.dm2gym import WhippingGym, make_vectorized_envs
from RL.ppo_continuous_action import Agent


def test(): # pylint: disable=missing-function-docstring
    """Module test"""
    env = WhippingGym("MultiStepTask")
    env.reset()
    env.step(env.action_space.sample())

    envs = make_vectorized_envs(num_envs=4,
                                asynchronous=True,
                                gamma=0.99,
                                env_id='MultiStepTask',
                                ctrl_type='torque',
                                fixed_time=False)
    envs.reset()
    out = envs.step(envs.action_space.sample())
    print(envs.observation_space, envs.action_space, out)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # agent = Agent(envs).to(device)
    # obs, _ = envs.reset(seed=0)
    # obs = torch.from_numpy(obs).float().to(device)
    # action = envs.action_space.sample()
    # action = torch.from_numpy(action).float().to(device)
    # agent(obs)
    # agent(obs, action)
    # print(envs.observation_space, np.array(envs.single_observation_space.shape).prod())
    # print(envs.single_action_space)
    import IPython; IPython.embed()
    # rand_actions = envs.action_space.sample()
    # envs.step(rand_actions)


if __name__ == '__main__':
    test()
