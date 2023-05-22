"""Module test"""
import numpy as np
import torch
from torchinfo import summary
from env.dm2gym import WhippingGym, make_vectorized_envs
from RL.ppo_continuous_action import Agent


def test(): # pylint: disable=missing-function-docstring
    """Module test"""
    env = WhippingGym("SingleStepTask-v0")
    env.reset()
    env.step(env.action_space.sample())

    envs = make_vectorized_envs(env_id="SingleStepTask-v0",
                                num_envs=4,
                                asynchronous=True,
                                gamma=0.99,)
    envs.reset(seed=0)
    print(envs.observation_space, np.array(envs.single_observation_space.shape).prod())
    rand_actions = envs.action_space.sample()
    envs.step(rand_actions)

    agent = Agent(envs, 32, "beta", True)
    tensor = torch.as_tensor(envs.single_observation_space.sample(), dtype=torch.float32).reshape(1, -1)
    import IPython; IPython.embed()


    return 0

if __name__ == '__main__':
    test()
