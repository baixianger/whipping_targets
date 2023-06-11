"""Environment module for the Whipping Targets project."""
from env_composer.arm import Arm
from env_composer.whip import Whip
from env_composer.target import Target
from env_composer.task import SingleStepTask
from env_composer.task import TwoStepTask
from env_composer.task import MultiStepTask
from env_composer.dm2gym import WhippingGym
