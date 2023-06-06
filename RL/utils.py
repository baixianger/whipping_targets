"""Utility for RL algorithms."""
from distutils.util import strtobool
from torch.utils.tensorboard import SummaryWriter
import wandb
from collections.abc import MutableMapping
# pylint: disable=too-many-arguments
# pylint: disable=line-too-long


def dict_flatten(cfg_dict, parent_key='', sep='.'):
    """Flatten a nested dictionary."""
    items = []
    for key, value in cfg_dict.items():
        new_key = parent_key + sep + key if parent_key else key
        if isinstance(value, dict):
            items.extend(dict_flatten(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    return dict(items)

def set_run_name(*args):
    """Set run name for wandb and tensorboard."""
    temp = [str(arg) if not isinstance(arg, str) else arg for arg in args]
    return "__".join(temp)

def set_track(wandb_project_name, wandb_entity, wandb_group, run_name, config, track):
    """Set wandb and tensorboard."""
    dict_cfg = flatten(config)
    if track:
        wandb.init(
            project=wandb_project_name,
            entity=wandb_entity,
            group=wandb_group,
            sync_tensorboard=True,
            config=dict_cfg,
            name=run_name,
            monitor_gym=True,
            save_code=True,
            )
    writer = SummaryWriter(f"runs/{run_name}")
    flat_cfg = dict_flatten(config)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in flat_cfg.items()])),
    )
    return writer

def flatten(dictionary, parent_key='', separator='_'):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)
