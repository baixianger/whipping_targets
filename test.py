"""Module test"""
import os
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
                                gamma=0.99,
                                ctrl_type="torque")
    envs.reset(seed=0)
    print(envs.observation_space, np.array(envs.single_observation_space.shape).prod())
    print(envs.single_action_space)
    rand_actions = envs.action_space.sample()
    envs.step(rand_actions)

    agent = Agent(envs, 32, "beta", True)
    tensor = torch.as_tensor(envs.single_observation_space.sample(), dtype=torch.float32).reshape(1, -1)
    freq = 2
    run_name = "test"
    for update in range(1, 10):
        freq = 2
        if update % freq == 0:
            torch.save(agent, f"checkpoints/{run_name}-update{update}.pth")
            # delete old checkpoints
            for filename in os.listdir("checkpoints"):
                print(filename, type(filename))
                old_update = update - freq
                if filename == f"{run_name}-update{old_update}.pth":
                    print(f"delete old checkpoint at update {old_update}.")
                    os.remove(f"checkpoints/{filename}")


    return 0

if __name__ == '__main__':
    test()
