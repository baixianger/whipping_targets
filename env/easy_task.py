"""Wrap all the entitys into a task, here it's to hit the target by the arm with a whip."""
import dataclasses
import numpy as np
# Composer high level imports
from dm_control import mjcf
from dm_control import composer
from dm_control.composer.observation import observable
from .task import RandomPos


# pylint: disable=invalid-name
# pylint: disable=unused-argument

@dataclasses.dataclass
class TaskRunningStats: # pylint: disable=too-many-instance-attributes
    """Running statistics for the task.
    """
    step_counter: int = 0
    time: int = 0
    time_buffer: list = dataclasses.field(default_factory=list)
    a2t: float = 3
    a2t_buffer: list = dataclasses.field(default_factory=list)
    w2t: float = 4
    w2t_buffer: list = dataclasses.field(default_factory=list)
    speed: float = 0
    speed_buffer: list = dataclasses.field(default_factory=list)
    old_a2t: float = 3
    old_w2t: float = 4
    old_speed: float = 0
    is_hitted: bool = False

    def reset(self):
        """Reset the statistics."""
        self.step_counter = 0
        self.time = 0
        self.time_buffer = []
        self.a2t = 3
        self.a2t_buffer = []
        self.w2t = 4
        self.w2t_buffer = []
        self.speed = 0
        self.speed_buffer = []
        self.old_a2t = 3
        self.old_w2t = 4
        self.old_speed = 0
        self.is_hitted = False

class Scene(composer.Entity):
    """A 7-DOF Panda arm."""

    def _build(self, *args, **kwargs):
        """Initializes the arm."""
        self._model = mjcf.from_path('env/xml/scene.xml')
        self._arm_joints = [self._model.find_all('joint')[i] for i in [0,1,2,3,4,5,6]]
        self._actuators = self._model.find_all('actuator')
        self._whip_start = self._model.find('body', 'whip_start')
        self._whip_end = self._model.find('body', 'whip_end')
        self._whip_bodies = [self._model.find('body', f"N{i:02d}") for i in range(27)]
        self._target = self._model.find('body', 'target')
        self._target_site = self._model.find('site', 'target_site')
        self._sensors = [
            self._model.sensor.add('framelinvel', name='target_vel', objtype='body', objname=self._target),
            self._model.sensor.add('framelinvel', name='whip_start_vel', objtype='body', objname=self._whip_start),
            self._model.sensor.add('framelinvel', name='whip_end_vel', objtype='body', objname=self._whip_end),
            self._model.sensor.add('touch', name='hit', site=self._target_site)]

    @property
    def mjcf_model(self):
        return self._model


class SingleStepTaskSimple(composer.Task):
    """basic task for whipping expriments"""
    def __init__(self, **kwargs):  # pylint: disable=too-many-arguments
        self.stats = TaskRunningStats()
        self.scene = Scene()

        self.arm_joints = self.scene._arm_joints
        self.whip_start = self.scene._whip_start
        self.whip_end = self.scene._whip_end
        self.whip_bodies = self.scene._whip_bodies
        self.target = self.scene._target
        self.target_site = self.scene._target_site
        self.actuators = self.scene._actuators
        self.sensors = self.scene._sensors
        
        self.num_substeps = 250
        self.max_steps = 1
        self.time_limit = 0.5
        self.set_timesteps(self.time_limit, 0.002)

        self._task_observables = {}
        self._task_observables['target'] = observable.MJCFFeature('xpos', self.target)
        # self._task_observables['whip_start'] = observable.MJCFFeature('xpos', self.whip_start)
        # self._task_observables['whip_end'] = observable.MJCFFeature('xpos', self.whip_end)
        # self._task_observables['whip_bodies'] = observable.MJCFFeature('xpos', self.whip_bodies)
        # self._task_observables['arm_qpos'] = observable.MJCFFeature('qpos', self.arm_joints)
        # self._task_observables['arm_qvel'] = observable.MJCFFeature('qvel', self.arm_joints)
        # self._task_observables['time'] = observable.Generic(lambda x: x.time())

        def _whip_start_to_target(physics):
            """Calculate the distance between arm and target."""
            return np.linalg.norm(
                physics.bind(self.whip_start).xpos - physics.bind(self.target).xpos)
        def _whip_end_to_target(physics):
            """Calculate the distance between whip end and target."""
            return np.linalg.norm(
                physics.bind(self.whip_end).xpos - physics.bind(self.target).xpos)

        for obs in self._task_observables.values():
            obs.enabled = True

        if kwargs.get('target', False):
            self.random_pos = RandomPos()

    @property
    def root_entity(self):
        return self.scene

    @property
    def task_observables(self):
        return self._task_observables

    def initialize_episode_mjcf(self, random_state):
        self.stats.reset()
        if hasattr(self, 'random_pos'):
            pos = self.random_pos()
            self.target.pos = pos

    def before_step(self, physics, action, random_state):
        physics.set_control(action)
        self.stats.w2t_buffer = []
        self.stats.a2t_buffer = []
        self.stats.speed_buffer = []

    def after_substep(self, physics, random_state):
        target_xpos = physics.bind(self.target).xpos
        whip_start_xpos = physics.bind(self.whip_start).xpos
        whip_end_xpos = physics.bind(self.whip_end).xpos
        self.stats.w2t_buffer.append(np.linalg.norm(target_xpos - whip_end_xpos))
        self.stats.a2t_buffer.append(np.linalg.norm(target_xpos - whip_start_xpos))
        speed = (physics.named.data.sensordata['target_vel'] - physics.named.data.sensordata['whip_end_vel'])\
                @ (target_xpos - whip_end_xpos) / np.linalg.norm(target_xpos - whip_end_xpos)
        self.stats.speed_buffer.append(speed)
        if physics.named.data.sensordata['hit'] > 1 and not self.stats.is_hitted:
            self.stats.is_hitted = True
            self.after_hit(physics)


    def after_step(self, physics, random_state):
        self.stats.w2t = np.min(self.stats.w2t_buffer)
        self.stats.a2t = np.min(self.stats.a2t_buffer)
        self.stats.speed = np.max(self.stats.speed_buffer)

    def should_terminate_episode(self, physics):
        return physics.time() > self.time_limit

    def get_reward(self, physics):
        """Reward is the sigmoid of the distance's reciprocal between the target and the whip end."""
        reward_w2t = 1 - self.stats.w2t
        reward_speed = self.stats.speed
        return reward_w2t
    
    def show_observables(self):
        """Show the observables."""
        for key, value in self.observables.items():
            print(f'{key:<30}', value)

    def after_hit(self, physics):
        """Change the color and size of the target."""
        self.target.geom[0].rgba = (0.96, 0.38, 0.08, 0.9)


