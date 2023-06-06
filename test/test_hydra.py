"""Test for PPO"""
# pylint: disable=import-error
# pylint: disable=line-too-long
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import IPython
from omegaconf import DictConfig, OmegaConf
import hydra
from collections.abc import MutableMapping

def flatten(dictionary, parent_key='', separator='_'):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def test(cfg: DictConfig):
    """Test for PPO"""
    print(OmegaConf.to_yaml(cfg))
    print(type(cfg.algo.target_kl))
    print(flatten(cfg))
    IPython.embed()




if __name__ == '__main__':
    test() # pylint: disable=no-value-for-parameter
