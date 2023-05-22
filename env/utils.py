"""Utility functions for Task Submodule."""
import dataclasses
from . import SingleStepTask, TwoStepTask

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
    "SingleStepTask-v0": SingleStepTask,
    "SingleStepTask-v1": SingleStepTask,
    "SingleStepTask-v2": SingleStepTask,
    "SingleStepTask-v3": SingleStepTask,
    "TwoStepTask-v0": TwoStepTask,
    "TwoStepTask-v1": TwoStepTask,
    "TwoStepTask-v2": TwoStepTask,
    "TwoStepTask-v3": TwoStepTask,
    "TwoStepTask-v4": TwoStepTask,
    "TwoStepTask-v5": TwoStepTask,
    "TwoStepTask-v6": TwoStepTask,
    "TwoStepTask-v7": TwoStepTask,
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
