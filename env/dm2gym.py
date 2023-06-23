"""Convert dm_control environment to gym environment."""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from dm_control import composer
from .easy_task import SingleStepTaskSimple, MultiStepTaskSimple
from .parabolic import ParabolicCascadeEnv, Viewer
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-arguments
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

task_list = {
    "SingleStepTaskSimple": SingleStepTaskSimple,
    "MultiStepTaskSimple": MultiStepTaskSimple,
    "ParabolicCascadeTask": ParabolicCascadeEnv,
    }

def register2gym(env_id, img_size=84, camera_id=0):
    """Register dm_control environment to gym interface.
    注册自定义gym环境, 但目前有问题是没法异步向量化."""
    if gym.envs.registry.get(env_id) is None:
        register(
            id=env_id,
            entry_point=WhippingGym,
            max_episode_steps=300,
            vector_entry_point=WhippingGym,
            kwargs={"env_id": env_id,
                    "img_size": img_size,
                    "camera_id": camera_id}
        )

def make_vectorized_envs(num_envs, asynchronous, **kwargs):
    """Set vectorized environment."""
    gym_env_fns = [lambda : make_gym_env(**kwargs) for _ in range(num_envs)]
    if asynchronous:
        return gym.vector.AsyncVectorEnv(gym_env_fns)
    return gym.vector.SyncVectorEnv(gym_env_fns, copy=False)

def make_gym_env(**kwargs):
    if kwargs.get("env_id") == "ParabolicCascadeTask":
        GymEnv = ParabolicCascadeEnv
    else:
        GymEnv = WhippingGym
    env = GymEnv(**kwargs)
    # env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
    env = gym.wrappers.RecordEpisodeStatistics(env)
    # env = gym.wrappers.ClipAction(env)
    # env = gym.wrappers.NormalizeObservation(env)
    # env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    # if kwargs.get("gamma"):
    #     env = gym.wrappers.NormalizeReward(env, gamma=kwargs["gamma"])
    # env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    return env

def make_dm_env(env_id, **kwargs):
    """Create a dm_control environment."""
    if task_list.get(env_id) is None:
        raise ValueError(f"Unsupported environment: {env_id}, \
                         it should be one of {task_list.keys()}.")
    dm_task = task_list[env_id](**kwargs)
    return dm_task, composer.Environment(task=dm_task)

class WhippingGym(gym.Env):
    """Convert dm_control environment to gym environment."""
    def __init__(self, env_id, img_size=84, camera_id=0, **kwargs):
        self.env_name = env_id
        self.img_size = img_size
        self.camera_id = camera_id
        self.task, self.env = make_dm_env(env_id, **kwargs)
        self.max_step = 200
        self.control_min = self.env.action_spec().minimum.astype(np.float32)
        self.control_max = self.env.action_spec().maximum.astype(np.float32)
        self.control_shape = self.env.action_spec().shape
        self._action_space = spaces.Box(
            self.control_min, self.control_max, self.control_shape)
        # Get observation space, flatten the observation
        total_size = 0 # qacc (7,) qpos (1, 7) 自定义的observation为空
        for _, value in self.env.observation_spec().items():
            total_size += np.array(value.shape).prod() if len(value.shape) > 0 else 1
        self._observation_space = spaces.Box(-np.inf, np.inf, (total_size, ))
        self.step_count = 0
        self.reward_range = (-np.inf, np.inf)
        self.metadata = {'render.modes': [
            'human', 'rgb_array'], 'video.frames_per_second': 30}

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def physics(self):
        return self.env.physics

    def reset(self, *, seed=None, options=None):
        self.seed(seed)
        obs = self.env.reset().observation
        obs_value = []
        for _, value in obs.items():
            obs_value.append(value.flatten())
        info = {}
        state = np.concatenate(obs_value).astype(np.float32)
        return state, info

    def step(self, action):
        obs = self.env.step(action)
        obs_value = []
        for _, value in obs.observation.items():
            obs_value.append(value.flatten())
        state = np.concatenate(obs_value).astype(np.float32)
        reward = obs.reward
        done = obs.step_type == 2
        truncated = False
        info = {}
        self.step_count += 1
        if done:
            self.step_count = 0
        return (state, reward, done, truncated, info)

    def render(self):
        height = width = self.img_size
        camera_id = self.camera_id
        if camera_id:
            img = self.env.physics.render(height, width, camera_id=camera_id)
        else:
            img = self.env.physics.render(height, width)
        return img

    def seed(self, seed):
        # pylint: disable=protected-access
        self.env._random_state = np.random.RandomState(seed)

    def close(self):
        pass
