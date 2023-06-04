"""Train different RL algorithms with different environments."""
from omegaconf import DictConfig
import hydra
# pylint: disable=import-error
# pylint: disable=line-too-long

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Train different RL algorithms with different environments."""
    if cfg.algo.name == 'ppo_continuous':
        from RL.ppo_continuous_action import trainer
    elif cfg.algo.name == 'ddpg_continuous':
        from RL.ddpg_continuous_action import trainer
    elif cfg.algo.name == 'sac_continuous':
        from RL.sac_continuous_action import trainer
    elif cfg.algo.name == 'td3_continuous':
        from RL.td3_continuous_action import trainer
    else:
        print('Algorithm not implemented yet.')

    trainer(cfg)

if __name__ == '__main__':
    main() # pylint: disable=no-value-for-parameter
