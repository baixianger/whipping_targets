"""Utility functions for Task Submodule."""
import dataclasses
import numpy as np
from dm_control.composer import variation
from dm_control.composer.variation import distributions
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from IPython.display import HTML

# Internal loading of video libraries.
# Font sizes
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def display_video(frames, framerate=30):
    height, width, _ = frames[0].shape
    dpi = 70
    orig_backend = matplotlib.get_backend()
    matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])
    def update(frame):
      im.set_data(frame)
      return [im]
    interval = 1000/framerate
    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                   interval=interval, blit=True, repeat=False)
    return HTML(anim.to_html5_video())


_HEIGHT_RANGE = (0.8, 1.0)
_RADIUS_RANGE = (0.8, 1.1)
_HEADING_RANGE = (-np.pi, np.pi)

_FIXED_ARM_QPOS = np.array([0, -1.76, 0, 0, 0, 3.75, 0]) # np.array([0, 0, 0, 0, 0, 0, 0])
_FIXED_TARGET_XPOS = np.array([0, 0, 0.5])

_RESET_QPOS = [
    np.array([ -3.9e-07,  -0.14,     3.9e-05, -1,       -0.00028,  2,       -0.52,     0.28,    -0.46,     0.16,
                -0.24,     0.087,   -0.12,     0.045,   -0.062,    0.023,   -0.032,    0.012,   -0.017,    0.0064,  
                -0.0088,   0.0033,  -0.0046,   0.0017,  -0.0023,   0.00082, -0.0011,   0.00034, -0.00046,  6.7e-05, 
                -9.4e-05, -8.4e-05,  0.00011, -0.00017,  0.00022, -0.00022,  0.00028, -0.00024,  0.0003,  -0.00025, 
                0.0003,  -0.00024,  0.00029, -0.00023,  0.00026, -0.0002,   0.00022, -0.00018,  0.00018, -0.00015, 
                0.00014, -0.00011,  0.0001,  -8e-05,    6.5e-05, -5.1e-05,  3.6e-05, -2.6e-05,  1.6e-05, -9.1e-06, 
                4.4e-06, ]),
    np.array([  1.3e-06,  0.66,    -0.00015, -0.073,    0.058,    2.4,      2.9,     -0.16,     0.76,    -0.12,
                0.41,    -0.07,     0.21,    -0.037,    0.11,    -0.02,     0.059,   -0.01,     0.032,   -0.0056,
                0.017,   -0.0031,   0.0099,  -0.0018,   0.0059,  -0.0011,    0.0038,  -0.00075,   0.0026,  -0.00057,
                0.002,   -0.00049,  0.0016,  -0.00045,  0.0015,  -0.00043,   0.0014,  -0.00043,   0.0013,  -0.00043,
                0.0012,  -0.00043,  0.0012,  -0.00043,  0.0011,  -0.00041,   0.00099, -0.00038,   0.00088, -0.00034,
                0.00075, -0.00029,  0.0006,  -0.00023,  0.00045, -0.00016,   0.0003,  -9.2e-05,   0.00017, -3.5e-05,
                6.1e-05, ])
    ]

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
