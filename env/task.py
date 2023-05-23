"""Wrap all the entitys into a task, here it's to hit the target by the arm with a whip."""
import dataclasses
import numpy as np
import mujoco
from dm_env import specs
# Composer high level imports
from dm_control import composer
from dm_control.composer.observation import observable
from dm_control.composer import variation

# Imports for Composer tutorial example
from dm_control.composer.variation import distributions
from dm_control.composer.variation import noises
from dm_control.locomotion.arenas import floors

from .arm import Arm
from .whip import Whip
from .target import Target

# pylint: disable=invalid-name
# pylint: disable=unused-argument

_HEIGHT_RANGE = (1, 1.5)
_RADIUS_RANGE = (1, 1.5)
_HEADING_RANGE = (-np.pi, np.pi)

_FIXED_ARM_QPOS = np.array([0, 0, 0, 0, 0, 0, 0])
_FIXED_TARGET_XPOS = np.array([0, 0, 0.5])

_CONTROL_TIMESTEP = 0.05  # control framerate should be 20Hz
_PHYSICS_TIMESTEP = 0.002  # physics framerate should be 500Hz


def sigmoid(x):  # pylint: disable=missing-function-docstring,invalid-name
    """Sigmoid function."""
    return 1 / (1 + np.exp(-x))

class FixedRandomPos():
    """A fixed target position generator."""
    # pylint: disable=too-many-arguments
    def __init__(self,
                 hight_range=_HEIGHT_RANGE,
                 radius_range=_RADIUS_RANGE,
                 heading_range=_HEADING_RANGE,
                 n=100, seed=42, **kwargs):
        if n == 0: # debug for RL algorithm only hit fixed target
            self.n = 1
            self.targets = np.array([[-1, 0, 1]])
        else:
            self.n = n
            self.targets = self.target_pos_generator(hight_range, radius_range, heading_range, n, seed)

    def target_pos_generator(self, hight_range, radius_range, heading_range, n, seed=42):
        """Generate a random target position. Shape is (n, 3)."""
        np.random.seed(seed) # only works inside the function scope
        iz = np.random.uniform(*hight_range, n)
        rad = np.random.uniform(*radius_range, n)
        theta = np.random.uniform(*heading_range, n)
        ix = rad * np.cos(theta)
        iy = rad * np.sin(theta)
        return np.vstack([ix, iy, iz]).T

    def __call__(self, random_state=None):
        np.random.seed(None)
        return self.targets[np.random.randint(self.n)]


class RandomPos(variation.Variation):  # pylint: disable=too-few-public-methods
    """A uniformly sampled position for the object."""

    def __init__(self,
                 hight_range=_HEIGHT_RANGE,
                 radius_range=_RADIUS_RANGE,
                 heading_range=_HEADING_RANGE,
                 **kwargs):
        self._height = distributions.Uniform(*hight_range)
        self._radius = distributions.Uniform(*radius_range)
        self._heading = distributions.Uniform(*heading_range)

    def __call__(self, initial_value=None, current_value=None, random_state=None):
        radius, heading, height = variation.evaluate(
            (self._radius, self._heading, self._height), random_state=random_state)
        return (radius*np.cos(heading), radius*np.sin(heading), height)


@dataclasses.dataclass
class TaskRunningStats: # pylint: disable=too-many-instance-attributes
    """Running statistics for the task.
    
    attributes:
        step_counter: int, the current step number taken in a episode
        timer: float, mujoco physics time at the start of current step
        time: int, the time when the target is hit
        distance: float, the aggregated target distance for current control step
        distance_buffer: list, the target distance list for current control step
        previous_time: int, the time when the target is hit in the previous control step
        previous_target_distance: float, the previous target distance
    methods:
        reset: reset the statistics
    
    """
    step_counter: int = 0
    timer: float = 0.0 # mujoco physics time at the start of current step
    time: int = float('inf')
    distance: float = float('inf')
    distance_buffer: list = dataclasses.field(default_factory=list)
    previous_time: int = 0
    previous_target_distance: float = float('inf')
    previous_whip_distance: float = 0

    def reset(self):
        """Reset the statistics."""
        self.step_counter = 0
        self.timer = 0.0
        self.time = float('inf')
        self.distance = float('inf')
        self.distance_buffer = []
        self.previous_time = 0
        self.previous_target_distance = float('inf')
        self.previous_whip_distance = 0


@dataclasses.dataclass
class TaskEntities:
    """Entities for the task.
    attributes:
        arm: Arm Entity object will be attached to the arena
        whip: Whip Entity object will be attached to the arm
        target: Target Entity object will be attached to the arena
        arena: Floor Entity object will host the arm and the target
    methods:
        install: install the entities into a single mjcf model
    """
    arm: Arm
    whip: Whip
    target: Target
    arena: floors.Floor

    def install(self):
        """Installs the entities into a single mjcf model."""
        self.arm.attach(self.whip, self.arm.end_effector_site)
        self.arena.attach(self.arm)
        self.arena.attach(self.target)

key_frame = np.array([ 1.94554078e-04,  2.14750126e-03,  3.39502991e-05, -3.30299255e-01,
                      -5.63442080e-04,  2.27793041e-02,  1.74814756e-03, -2.01016613e-01,
                       2.35287420e-02, -1.10745372e-01,  1.33430657e-02, -6.33656953e-02,
                       8.28183647e-03, -3.80109847e-02,  6.12615765e-03, -2.52217774e-02,
                       5.24265351e-03, -1.67522289e-02,  3.33613481e-03, -1.36889706e-02,
                       2.07464681e-03, -1.37121126e-02,  3.53944464e-03, -1.34581836e-02,
                       4.52296711e-03, -1.28308707e-02,  5.17379965e-03, -1.38731480e-02,
                       7.20811993e-03, -1.49824154e-02,  8.28500445e-03, -1.61220593e-02,
                       8.70748457e-03, -1.75925399e-02,  9.52036647e-03, -1.90088823e-02,
                       1.05211317e-02, -1.93481261e-02,  1.10960942e-02, -1.73823887e-02,
                       1.05858611e-02, -1.15217481e-02,  8.33064990e-03,  1.87467557e-04,
                       3.61796101e-03,  1.97248685e-02, -4.49516326e-03,  4.84197498e-02,
                      -1.66752753e-02,  8.67328836e-02, -3.34145100e-02,  1.34580639e-01,
                      -5.36959499e-02,  1.88451865e-01, -7.53259207e-02,  2.40644234e-01,
                      -9.43795758e-02,  2.79265676e-01, -1.05867951e-01,  2.90582510e-01,
                      -1.05755340e-01,  2.64114859e-01, -9.27365854e-02,  1.97256521e-01,
                      -6.80445888e-02,  9.84576973e-02, -3.48248498e-02,  8.55738387e-01,
                       1.08593278e-01,  5.04003407e-01, -4.35876025e-02])

class _BasicTask(composer.Task):
    """basic task for whipping expriments
    
    Attributes:
        (properties)root_entity: the root entity of the task
        (properties)observables: dict, observables for the task
        (properties)task_observables: dict, observables for the task
        (properties)control_timestep: float, control timestep
        (properties)physics_timestep: float, physics timestep
        (properties)physics_steps_per_control_step: int, physics steps per control step
        stats: TaskRunningStats, running statistics for the task
        entities: TaskEntities, entities for the task
        _mjcf_variator: MJCFVariator, variator for the mjcf model
        _physics_variator: PhysicsVariator, variator for the physics
        _task_observables: dict, observables for the task
        _initial_arm_qpos: defaut np.array([0, 0, 0, 0, 0, 1.5, 0]), initial arm qpos
        _target_pos: lambda callable, target position generator, 
                    if None, use RandomPos, otherwise randomly choose one from the given target list
    Methods:
        (父)set_timesteps: set the control timestep and the physics timestep
        (父)action_spec: return the action spec
        (父)reset: reset the task
        (父)step: take a step in the task
        (父)initialize_episode_mjcf: initialize the mjcf model
        (父)initialize_episode: initialize the physics
        (父)before_step: update the running statistics before each step
        (父)before_substep: update the running statistics before each substep
        (父)after_substep: update the running statistics after each substep
        (父)after_step: update the running statistics after each step
        (父)get_reward: get the reward for the current step
        (父)should_terminate_episode: check if the episode should be terminated

        show_observables: show the observables
        _initialize_arm_qpos: initialize the arm qpos
        _initialize_target_xpos: initialize the target xpos
        _observables_config: enable the observables needed and add noise/corruptor for subclass
        _set_noise: set the noise for the observables
        _distance_w2t: calculate the distance between the whip end and the target
        _distance_b2e: calculate the distance between the whip begin and the whip end
        _hit_detection: detect if the whip end hits the target
        _reward_target_close/_reward_whip_away/_reward_main_time: Supplemental Awards
        """

    def __init__(self,
                 ctrl_type='position',
                 whip_type=0,
                 target=None,
                 obs_noise=None, # 0.01
                 **kwargs
                 ):  # pylint: disable=too-many-arguments
        self.stats = TaskRunningStats()

        self.entities = TaskEntities(arm=Arm(ctrl_type),
                                     whip=Whip(whip_type),
                                     target=Target(),
                                     arena=floors.Floor(),)
        self.entities.install()
        self.all_joints = self.root_entity.mjcf_model.find_all("joint")
        self._mjcf_variator = variation.MJCFVariator()
        self._physics_variator = variation.PhysicsVariator()

        self._task_observables = {}
        self._task_observables['time'] = observable.Generic(lambda x: self.stats.time)
        self._task_observables['distance'] = observable.Generic(lambda x: self.stats.distance)

        if obs_noise is not None:
            self._set_noise(obs_noise)

        if target is not None and isinstance(target, int):
            self._target_pos = FixedRandomPos(**kwargs, n=target)
        else:
            self._target_pos = RandomPos(**kwargs)

    @property
    def root_entity(self):
        return self.entities.arena

    @property
    def task_observables(self):
        return self._task_observables

    def initialize_episode_mjcf(self, random_state):
        self._mjcf_variator.apply_variations(random_state)
        self.stats.reset()

    def initialize_episode(self, physics, random_state):
        self._physics_variator.apply_variations(physics, random_state)
        physics.bind(self.all_joints).qpos = key_frame
        self.entities.target.set_pose(physics, self._target_pos(random_state))

    def before_step(self, physics, action, random_state):
        self.stats.distance_buffer = []
        self.stats.timer = physics.time()
        physics.set_control(action)

    def after_substep(self, physics, random_state):
        vector = self._whip_to_target(physics)
        distance = np.linalg.norm(vector)
        self.stats.distance_buffer.append(distance)

    def after_step(self, physics, random_state):
        self.stats.step_counter += 1
        self.stats.distance = np.min(self.stats.distance_buffer)

    def get_reward(self, physics):
        return 1.0

    # ----------自定义函数----------
    def _observables_config(self, names):
        """Enable the observables needed and add noise/corruptor for subclass."""
        for key in names:
            setattr(self.observables[key], 'enabled', True)

    def _set_noise(self, obs_noise):
        norm_corrptor = noises.Additive(distributions.Normal(scale=obs_noise))
        log_corruptor = noises.Multiplicative(
            distributions.LogNormal(sigma=obs_noise))
        self.entities.arm.observables.arm_joints_qpos.corruptor = norm_corrptor
        self.entities.arm.observables.arm_joints_qvel.corruptor = log_corruptor
        self.entities.arm.observables.arm_joints_qacc.corruptor = log_corruptor
        self.entities.arm.observables.arm_joints_qfrc.corruptor = log_corruptor
        self.entities.whip.observables.whip_begin_xpos.corruptor = norm_corrptor
        self.entities.whip.observables.whip_end_xpos.corruptor = norm_corrptor
        self.entities.target.observables.target_xpos.corruptor = norm_corrptor
        self._task_observables['time'].corruptor = norm_corrptor
        self._task_observables['distance'].corruptor = norm_corrptor

    def _whip_to_target(self, physics):  # w2t: whip to target
        # 一个内置的计算方法
        # self.entities.target.global_vector_to_local_frame(physics, whip_pos)
        source = physics.bind(self.entities.whip.whip_end).xpos
        target = physics.bind(self.entities.target.target_body).xpos
        return np.linalg.norm(source - target)

    def _hit_detection(self, physics):
        target = self.entities.target.target_body
        source = self.entities.whip.whip_end
        source_id = physics.bind(source).element_id
        target_id = physics.bind(target).element_id
        contact_pairs = zip(physics.data.contact.geom1,
                            physics.data.contact.geom2)
        return ((source_id, target_id) in contact_pairs)\
            or ((target_id, source_id) in contact_pairs)

    def show_observables(self):
        """Show the observables."""
        for key, value in self.observables.items():
            print(f'{key:<30}', value)


class SingleStepTask(_BasicTask):
    """Under a fixed initial position, whip the target in one step.

    action:      control of the arm joints, 7-dim action space
    state:       状态和观测区别不是很大, 如果环境是完全可观测的, 那么状态和观测就是一样的
    observation: arm joints qpos qacc
    reward:      sigmoid of the distance's reciprocal between the target and the whip end
    """

    def __init__(self,
                 ctrl_type='position',
                 whip_type=0,
                 target=None,
                 obs_noise=None,
                 **kwargs,
                 ):  # pylint: disable=too-many-arguments
        super().__init__(ctrl_type, whip_type, target, obs_noise, **kwargs)
        self.time_limit = 1
        self.max_steps = 1
        self.set_timesteps(1, 0.002)
        self._observables_config(['arm/arm_joints_qpos',
                                  'arm/whip/whip_end_xpos',
                                  'target/target_xpos'])

    def should_terminate_episode(self, physics):
        return physics.time() > self.time_limit

    def get_reward(self, physics):
        return 10 ** -self.stats.distance


class TwoStepTask(_BasicTask):
    """A specific task that only using two position control steps.

    first step is under the given target to choose a optimized get-ready position
    another step is to whip the target.
    """
    def __init__(self,
                 ctrl_type='position',
                 whip_type=0,
                 target=None,
                 obs_noise=None,
                 fixed_time=True,
                 **kwargs
                 ):  # pylint: disable=too-many-arguments
        super().__init__(ctrl_type, whip_type, target, obs_noise, **kwargs)
        self._initial_arm_qpos = np.array([0, 0, 0, 0, 0, 1.5, 0])
        self._target_pos = RandomPos(
            (1, 1.5), (1, 1.5), (-5/4 * np.pi, -3/4 * np.pi))
        self._time_limit = 1
        self._max_steps = 2
        self.set_timesteps(0.5, 0.002)
        self._observables_config(['arm/arm_joints_qpos', 'arm/arm_joints_qacc',
                                  'time', 'distance'])
        self._fixed_time = fixed_time

    def before_step(self, physics, action, random_state):
        self.stats.distance_buffer = []
        self.stats.timer = physics.time()
        if not self._fixed_time:
            n_ratio = int(action[-1] / self.physics_timestep)
            self.control_timestep = n_ratio * self.physics_timestep
            physics.set_control(action[:-1])
        else:
            physics.set_control(action)

    def after_substep(self, physics, random_state):
        distance = self._whip_to_target(physics)
        self.stats.distance_buffer.append(distance)

    def after_step(self, physics, random_state):
        self.stats.step_counter += 1
        if self.stats.step_counter == 1: # 第一步取最远距离
            self.stats.distance = np.max(self.stats.distance_buffer)
        if self.stats.step_counter == 2: # 第二部取最短距离
            idx = np.argmin(self.stats.distance_buffer)
            self.stats.distance = self.stats.distance_buffer[idx]
            if self.stats.distance <= 0.1:
                self.stats.time = (idx+1) * self.physics_timestep + self.stats.timer

    def should_terminate_episode(self, physics):
        return self.stats.step_counter >= 2 or physics.time() > self._time_limit

    def get_reward(self, physics):
        if self.stats.distance >= 2.5 and self.stats.step_counter == 1:
            return np.clip(3.5 - self.stats.distance, a_min=0, a_max=1)
        if self.stats.distance <= 1 and self.stats.step_counter == 2:
            return 1.0 - self.stats.distance
        return 0.0

    def action_spec(self, physics):
        names = [physics.model.id2name(i, 'actuator') or str(i)
                for i in range(physics.model.nu)]
        num_actions = physics.model.nu
        is_limited = physics.model.actuator_ctrllimited.ravel().astype(bool)
        control_range = physics.model.actuator_ctrlrange
        minima = np.full(num_actions, fill_value=-mujoco.mjMAXVAL, dtype=np.float32)
        maxima = np.full(num_actions, fill_value=mujoco.mjMAXVAL, dtype=np.float32)
        minima[is_limited], maxima[is_limited] = control_range[is_limited].T
        if not self._fixed_time:
            names.append('control_time')
            num_actions += 1
            minima = np.append(minima, 0)
            maxima = np.append(maxima, 0.5)
        return specs.BoundedArray(shape=(num_actions,),
                                dtype=np.float32,
                                minimum=minima,
                                maximum=maxima,
                                name='\t'.join(names))
