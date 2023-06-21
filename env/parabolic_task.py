"""Wrap all the entitys into a task, here it's to hit a parabolic target by the arm with a whip."""
import dataclasses
import numpy as np
# Composer high level imports
import torch
import mujoco
from dm_control import mjcf
from dm_control import composer
from dm_control.utils import containers
from dm_control.composer.observation import observable
from.utils import FixedRandomPos, RandomPos, TaskRunningStats, _RESET_QPOS

# pylint: disable=invalid-name
# pylint: disable=unused-argument


class ParabolicTaskCascade():
    """basic task for whipping expriments"""
    def __init__(self, **kwargs):  # pylint: disable=too-many-arguments
        self.arm_qpos = _RESET_QPOS[1]
        self.model = mujoco.MjModel.from_xml_path("env/xml/scene_parabola_position.xml")
        self.data = mujoco.MjData(self.model)
        self.mocap = np.load('env/xml/target_mocap.npy')
        self.agent = torch.load("checkpoints/单步环境(位置控制和正态分布)+PPO+奖励研究/ppo_single_reward5_update300.pth", 
                                map_location=torch.device("cpu")) # 在 cpu 上进行推理，避免拷贝数据到 gpu

    def arm_ctrl_pred(self, target_pos):
        with torch.no_grad():
            x = torch.Tensor(target_pos).view(1, -1)
            ctrl, _, _ = self.agent.policy(x)
        return ctrl[0].numpy()

    def arm_ctrl_exec(self, arm_action):
        """control the arm"""
        self.data.ctrl[:] = arm_action

    def trajecotry_pred(self, target_pos_list):
        """dummy trajectory prediction"""
        idx = np.searchsorted(self.mocap[:, 2], target_pos_list[-1][1])
        time_pred = self.mocap[idx][0]
        target_pos_pred = np.array([-0.9, 0, 1])
        return time_pred, target_pos_pred

    def reset(self):
        random_idx = np.random.randint(0, len(self.mocap)-150)
        qpos = np.array([*self.mocap_data[random_idx][1:4], 1, 0, 0, 0])
        qvel = np.array([*self.mocap_[random_idx][4:], 0, 0, 0])
        self.data.qpos[:-7] = _RESET_QPOS[1]
        self.data.joint('target').qpos = qpos
        self.data.joint('target').qvel = qvel
        target_pos_list = []

        while qpos[1] <= -6: # 保证目标点在摄像机视野内
            mujoco.mj_forward(self.model, self.data)
            if qpos[1] > -6: break

        for _ in range(10): # 收集足够的序列进行轨迹预测
            mujoco.mj_forward(self.model, self.data)
            target_pos_list.append(self.data.joint('target').qpos)
        # 因为如果进行实时轨迹预测，会导致模型的推理时间过长，同时也设计异步操作，
        # 所以这里采用一次性预测的方式，预测完后不进行修正
        time_pred, target_pos_pred = self.trajecotry_pred(target_pos_list)
        return time_pred + self.data.time, target_pos_pred

    def step(self, action):
        # time是绝对时间不是相对时间
        delta_T = action[0] if action[0] > 0 else 0
        for _ in range(delta_T//self.model.opt.timestep):
            mujoco.mj_step(self.model, self.data)

        ctrl = self.arm_ctrl_pred(action[1:])
        self.arm_ctrl_exec(ctrl)
        reward = self.get_reward(self.data)
        return obs, reward, True, False, {}
            
    
    def get_reward(self, data):
        # keep sampling for 0.4s
        target_buffer = []
        whip_buffer = []
        sensor_buffer = []
        for _ in range(0.4 // self.model.opt.timestep):
            mujoco.mj_step(self.model, self.data)
            target_buffer.append(data.body_xpos['target'])
            whip_buffer.append(data.body_xpos['whip_end'])
            sensor_buffer.append(data.sensordata['hit'])

        if self.reward_type == 0:
            reward = 1 if np.max(sensor_buffer) > 1 else 0
        if self.reward_type == 1:
            w2t_distance = np.linalg.norm(np.array(target_buffer) - np.array(whip_buffer), axis=-1, keepdims=True).min()
            reward = w2t_distance
        return reward
                

        


