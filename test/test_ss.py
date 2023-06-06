"""Module test"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from dm_control import composer
import gymnasium as gym
from env.dm2gym import WhippingGym, make_vectorized_envs
from env.easy_task import SingleStepTaskSimple

def test(): # pylint: disable=missing-function-docstring
    """Module test"""
    
    ## DM Control
    dm_task = SingleStepTaskSimple(target=False)
    dm_env = composer.Environment(dm_task)
    dm_env.reset()
    physics = dm_env.physics

    # OpenAI Gym
    gym_env = WhippingGym("SingleStepTaskSimple", target=False)
    gym_env = gym.wrappers.RecordEpisodeStatistics(gym_env)
    gym_env.reset()
    gym_env.step(gym_env.action_space.sample())

    # Vectorized OpenAI Gym
    envs = make_vectorized_envs(num_envs=2,
                                asynchronous=True,
                                env_id="SingleStepTaskSimple",
                                target=False,)
    envs.reset(seed=0)
    actions = envs.action_space.sample() 
    envs.step(actions)

    # Run bash command
    os.system("echo $HOSTNAME")
    
    import IPython; IPython.embed()


if __name__ == '__main__':
    test()
