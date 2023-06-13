"""Test for PPO"""
# pylint: disable=import-error
# pylint: disable=line-too-long
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import IPython
import numpy as np
import envpool
from dm_control import mjcf
from dm_control import viewer
from dm_control import composer
from dm_control import suite
from env.easy_task import SingleStepTaskSimple, MultiStepTaskSimple


def register_suite():
    config_dict = {'ctrl_type': 'position',
                   'arm_pos': 1,
                   'target': 0,}
    env = suite.load('easy_task', 'single_step', environment_kwargs=config_dict)
    # envpool.list_all_envs()
    # envpool_env = envpool.make_dm("EasyTaskSingleStep", num_envs=2, environment_kwargs=config_dict)
    IPython.embed()



if __name__ == '__main__':
    register_suite() # pylint: disable=no-value-for-parameter
