"""Module test"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from dm_control import composer
from env.dm2gym import WhippingGym, make_vectorized_envs
from env.easy_task import SingleStepTaskSimple

def test(): # pylint: disable=missing-function-docstring
    """Module test"""
    
    ## DM Control
    dm_task = SingleStepTaskSimple(target=True)
    dm_env = composer.Environment(dm_task)
    dm_env.reset()
    physics = dm_env.physics

    # OpenAI Gym
    gym_env = WhippingGym("SingleStepTaskSimple", target=True)
    gym_env.reset()
    gym_env.step(gym_env.action_space.sample())

    # Vectorized OpenAI Gym
    envs = make_vectorized_envs(num_envs=4,
                                asynchronous=True,
                                env_id="SingleStepTaskSimple",
                                target=True,)
    envs.reset(seed=0)

    # Run bash command
    os.system("echo $HOSTNAME")
    
    import IPython; IPython.embed()


if __name__ == '__main__':
    test()
