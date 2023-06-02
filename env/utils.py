"""Utility functions for Task Submodule."""
import dataclasses
from .task import SingleStepTask, TwoStepTask, MultiStepTask
from .easy_task import SingleStepTaskSimple
# Graphics-related
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



# | Task Name     |Version|Control Method|Whip Type|Target Mode|Noise| Action  |
# |:-------------:|:-----:|:------------:|:-------:|:---------:|:---:|:-------:|
# |SingleStepTask |v0     |Position      |0:MIT    |100 points |w/o  |fixed0.5s|
# |SingleStepTask |v1     |Position      |1:Mujoco |100 points |w/o  |fixed0.5s|
# |SingleStepTask |v3     |Torque        |0:MIT    |100 points |w/o  |fixed0.5s|
# |SingleStepTask |v4     |Torque        |1:Mujoco |100 points |w/o  |fixed0.5s|
# |TwoStepTask    |v0     |Position      |0:MIT    |100 points |w/o  |fixed0.5s|
# |TwoStepTask    |v1     |Position      |0:MIT    |100 points |w/o  |unfixed  |
# |TwoStepTask    |v2     |Position      |1:Mujoco |100 points |w/o  |fixed0.5s|
# |TwoStepTask    |v3     |Position      |1:Mujoco |100 points |w/o  |unfixed  |
# |TwoStepTask    |v4     |Torque        |0:MIT    |100 points |w/o  |fixed0.5s|
# |TwoStepTask    |v5     |Torque        |0:MIT    |100 points |w/o  |unfixed  |
# |TwoStepTask    |v6     |Torque        |1:Mujoco |100 points |w/o  |fixed0.5s|
# |TwoStepTask    |v7     |Torque        |1:Mujoco |100 points |w/o  |unfixed  |

@dataclasses.dataclass
class TaskDict: # pylint: disable=too-many-ancestors
    """Task dictionary for the Whipping Targets project."""
    task_list = {
    "SingleStepTask": SingleStepTask,
    "TwoStepTask": TwoStepTask,
    "MultiStepTask": MultiStepTask,
    "SingleStepTaskSimple": SingleStepTaskSimple,
    }

    task_setting = {
"SingleStepTask-v0": {"ctrl_type": "poistion", "whip_type": 0, "target": None},
"SingleStepTask-v1": {"ctrl_type": "poistion", "whip_type": 1, "target": None},
"SingleStepTask-v2": {"ctrl_type": "torque", "whip_type": 0, "target": None},
"SingleStepTask-v3": {"ctrl_type": "torque", "whip_type": 1, "target": None},
"TwoStepTask-v0": {"ctrl_type": "poistion", "whip_type": 0, "target": None, "fixed_time": True},
"TwoStepTask-v1": {"ctrl_type": "poistion", "whip_type": 0, "target": None, "fixed_time": False},
"TwoStepTask-v2": {"ctrl_type": "poistion", "whip_type": 1, "target": None, "fixed_time": True},
"TwoStepTask-v3": {"ctrl_type": "poistion", "whip_type": 1, "target": None, "fixed_time": False},
"TwoStepTask-v4": {"ctrl_type": "torque", "whip_type": 0, "target": None, "fixed_time": True},
"TwoStepTask-v5": {"ctrl_type": "torque", "whip_type": 0, "target": None, "fixed_time": False},
"TwoStepTask-v6": {"ctrl_type": "torque", "whip_type": 1, "target": None, "fixed_time": True},
"TwoStepTask-v7": {"ctrl_type": "torque", "whip_type": 1, "target": None, "fixed_time": False},
}
