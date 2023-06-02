"""Environment module for the Whipping Targets project."""
from env.arm import Arm
from env.whip import Whip
from env.target import Target
from env.task import SingleStepTask
from env.task import TwoStepTask
from env.task import MultiStepTask
from env.dm2gym import WhippingGym
from env.easy_task import SingleStepTaskSimple
