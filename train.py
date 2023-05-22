"""Train different RL algorithms with different environments."""
from omegaconf import DictConfig
import hydra
from RL.ppo_continuous_action import trainer
# pylint: disable=import-error
# pylint: disable=line-too-long

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Train different RL algorithms with different environments."""
    trainer(cfg)

if __name__ == '__main__':
    main() # pylint: disable=no-value-for-parameter
