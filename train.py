"""Train different RL algorithms with different environments."""
from omegaconf import DictConfig
import hydra
# pylint: disable=import-error
# pylint: disable=line-too-long

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Train different RL algorithms with different environments."""
    if cfg.algo.name == 'ppo':
        from RL.ppo_continuous_action import trainer
    elif cfg.algo.name == 'ddpg':
        from RL.ddpg_continuous_action import trainer
    elif cfg.algo.name == 'sac':
        from RL.sac_continuous_action import trainer
    elif cfg.algo.name == 'td3':
        from RL.td3_continuous_action import trainer
    else:
        print('Algorithm not implemented yet.')
        return 0

    trainer(cfg)

if __name__ == '__main__':
    main() # pylint: disable=no-value-for-parameter
