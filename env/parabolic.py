"""Wrap all the entitys into a task, here it's to hit a parabolic target by the arm with a whip."""
import dataclasses
from typing import Any
import numpy as np
# Composer high level imports
import torch
import mujoco
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from mujoco.glfw import glfw

# pylint: disable=invalid-name
# pylint: disable=unused-argument


class ParabolicCascadeEnv(gym.Env):
    """basic task for whipping expriments"""
    def __init__(self, **kwargs):  # pylint: disable=too-many-arguments
        self.model = mujoco.MjModel.from_xml_path("env/xml/scene_parabola_position.xml")
        self.data = mujoco.MjData(self.model)
        self.mocap = np.load('env/xml/target_mocap.npy')
        self.arm_ctrl_model = torch.load("checkpoints/单步环境(位置控制和正态分布)+PPO+奖励研究/ppo_single_reward5_update300.pth", 
                                map_location=torch.device("cpu")) # 在 cpu 上进行推理，避免拷贝数据到 gpu
        self.trajecotry_model = None
        self.hit_pos = None
        
        # 继承自 gym.Env 的属性
        self.metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 30}
        self.max_step = 2000 # doesn't matter
        self.action_space = spaces.Box(np.array([0,]), np.array([np.inf]), (1,))
        self.observation_space = spaces.Box(-np.inf, np.inf, (4, ))
        self.reward_range = (-np.inf, np.inf)

    def reset(self, *, seed=None, options=None):
        """重置环境，返回初始状态。返回的状态为到达预测击中点的时间和位置
        !!! 时间为模拟器运行的绝对时间，即估计的是在何时到达预测点而不是剩余多少时间到达预测点"""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        random_idx = np.random.randint(0, len(self.mocap)-250)
        qpos = np.array([*self.mocap[random_idx][1:4], 1, 0, 0, 0])
        qvel = np.array([*self.mocap[random_idx][4:], 0, 0, 0])
        self.data.joint('target').qpos = qpos
        self.data.joint('target').qvel = qvel
        target_pos_seq = []

        while qpos[1] <= -6: # 保证目标点在摄像机视野内, 6米外空转等待
            mujoco.mj_step(self.model, self.data)

        for _ in range(10): # 收集足够的序列进行轨迹预测
            mujoco.mj_step(self.model, self.data)
            target_pos_seq.append(self.data.joint('target').qpos)
        # 因为如果进行实时轨迹预测，会导致模型的推理时间过长，同时也设计异步操作，
        # 所以这里采用一次性预测的方式，预测完后不进行修正
        target_pos_seq = torch.Tensor(target_pos_seq).view(1, -1, 7)
        obs = self.hit_pred(target_pos_seq) # hit_time and hit_pos
        return obs, {}

    def step(self, action):
        """输入的动作参数是时间，单位为秒，模拟器会运行到该时间，然后开始抽打
        !!! 时间为模拟器运行的绝对时间，即估计的是在何时控制抽打而不是剩余多少时间控制抽打"""
        whip_time = action if action > 0 else 0
        while whip_time > self.data.time:
            mujoco.mj_step(self.model, self.data)
        reward = self.start_whippinig()
        return [0, 0, 0, 0] , reward, True, False, {}

    def hit_pred(self, target_pos_seq):
        """dummy trajectory prediction"""
        idx = np.searchsorted(self.mocap[:, 2], target_pos_seq[0][-1][1])
        time_pred = self.mocap[idx][0] # 从当下帧开始多久后到达预定目标位，相对时间
        self.hit_pos = np.array([[-0.9, 0, 1]])[0] # batch * 3
        return (time_pred + self.data.time, *self.hit_pos)
                
    def start_whippinig(self):
        with torch.no_grad():
            x = torch.Tensor(self.hit_pos).view(1, -1)
            ctrl, _, _ = self.arm_ctrl_model.policy(x)
        self.data.ctrl[:] = ctrl[0].numpy()
        # keep sampling for 0.4s
        target_buffer = []
        whip_buffer = []
        sensor_buffer = []
        for _ in range(int(0.4 /self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)
            target_buffer.append(self.data.body('target').xpos)
            whip_buffer.append(self.data.body('whip_end').xpos)
            sensor_buffer.append(self.data.sensordata[0]) # alter: self.data.sensor("hit").data[0]
            if self.data.joint("target").qpos[2] > 0.1 and self.data.sensordata[0] > 1:
                return 1.0
        w2t_distance = np.linalg.norm(np.array(target_buffer) - np.array(whip_buffer), axis=-1, keepdims=True).min()
        reward = 1 - w2t_distance
        return reward
    
    def viewer(self):
        pass

    def close(self):
        pass


class Viewer:
    def __init__(self, env, agent):
        self.button_left = False
        self.button_middle = False
        self.button_right = False
        self.lastx = 0
        self.lasty = 0
        self.model = env.model
        self.data = env.data
        self.cam = mujoco.MjvCamera()                        # Abstract camera
        self.opt = mujoco.MjvOption()                        # Visualization options

        # Init GLFW, create window, make OpenGL context current, request v-sync
        glfw.init()
        self.window = glfw.create_window(1200, 900, "Demo", None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        # initialize visualization data structures
        mujoco.mjv_defaultCamera(self.cam)
        mujoco.mjv_defaultOption(self.opt)
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

        # install GLFW mouse and keyboard callbacks
        glfw.set_key_callback(self.window, self.keyboard)
        glfw.set_cursor_pos_callback(self.window, self.mouse_move)
        glfw.set_mouse_button_callback(self.window, self.mouse_button)
        glfw.set_scroll_callback(self.window, self.scroll)

        # self.cam.azimuth = 90 ; self.cam.elevation = -89 ; self.cam.distance = 11
        # self.cam.lookat =np.array([ 0.0 , 0.0 , 0.0 ])

    def keyboard(self, window, key, scancode, act, mods):
        if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
            mujoco.mj_resetDataKeyFrame(self.model, self.data, 0)
            mujoco.mj_forward(self.model, self.data)

    def mouse_button(self, window, button, act, mods):
        self.button_left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        self.button_middle = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        self.button_right = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        # update mouse position
        glfw.get_cursor_pos(window)

    def mouse_move(self, window, xpos, ypos):
        # compute mouse displacement, save
        dx = xpos - self.lastx
        dy = ypos - self.lasty
        self.lastx = xpos
        self.lasty = ypos

        # no buttons down: nothing to do
        if (not self.button_left) and (not self.button_middle) and (not self.button_right):
            return

        # get current window size
        width, height = glfw.get_window_size(window)

        # get shift key state
        PRESS_LEFT_SHIFT = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
        PRESS_RIGHT_SHIFT = glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

        # determine action based on mouse button
        if self.button_right:
            if mod_shift:
                action = mujoco.mjtMouse.mjMOUSE_MOVE_H
            else:
                action = mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif self.button_left:
            if mod_shift:
                action = mujoco.mjtMouse.mjMOUSE_ROTATE_H
            else:
                action = mujoco.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM

        mujoco.mjv_moveCamera(self.model, action, dx/height,
                              dy/height, self.scene, self.cam)
        
    def scroll(self, window, xoffset, yoffset):
        action = mujoco.mjtMouse.mjMOUSE_ZOOM
        mujoco.mjv_moveCamera(self.model, action, 0.0, -0.05 * yoffset, self.scene, self.cam)

    def viewer(self, show="notebook"):
        while not glfw.window_should_close(self.window):
            time_prev = self.data.time

            while (self.data.time - time_prev < 1.0/60.0):
                mujoco.mj_step(self.model, self.data)

            if (self.data.time>=2):
                break;

            # get framebuffer viewport
            viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
            viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)

            #print camera configuration (help to initialize the view)
            # print('cam.azimuth =',self.cam.azimuth,';','cam.elevation =',self.cam.elevation,';','cam.distance = ',self.cam.distance)
            # print('cam.lookat =np.array([',self.cam.lookat[0],',',self.cam.lookat[1],',',self.cam.lookat[2],'])')

            # Update scene and render
            mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                            mujoco.mjtCatBit.mjCAT_ALL.value, self.scene)
            mujoco.mjr_render(viewport, self.scene, self.context)

            # swap OpenGL buffers (blocking call due to v-sync)
            glfw.swap_buffers(self.window)

            # process pending GUI events, call GLFW callbacks
            glfw.poll_events()

        glfw.terminate()