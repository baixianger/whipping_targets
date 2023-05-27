"""Test for PPO"""
# pylint: disable=import-error
# pylint: disable=line-too-long
import IPython
from omegaconf import DictConfig, OmegaConf
import hydra
import numpy as np
from RL.ppo_continuous_action import Agent
from env.dm2gym import make_vectorized_envs


@hydra.main(version_base=None, config_path="conf", config_name="config")
def test(cfg: DictConfig):
    """Test for PPO"""
    print(OmegaConf.to_yaml(cfg))
    print(type(cfg.algo.target_kl))

    
    hidden_dims = cfg.algo.hidden_dims
    envs = make_vectorized_envs(env_id="SingleStepTask-v0",
                                num_envs=4,
                                asynchronous=True,
                                gamma=0.99,
                                ctrl_type="torque")
    # agent = Agent(envs, hidden_dims, cfg.algo.action_dist)

    from torch import nn
    def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        """Trailed orthogonal weight initialization.
        For the policy network: initialization of weights with scaling 1.414 and 0.01.
        For the value network: initialization of weights with scaling 1.414 and 1.
        """
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer
    def head(in_features, hidden_dims, init_func=layer_init, **kwargs):
        """Create a head template for actor and critic (aka. Agent network)"""
        layers = []
        for in_dim, out_dim in zip((in_features,)+hidden_dims, hidden_dims):
            layers.append(init_func(nn.Linear(in_dim, out_dim), **kwargs))
            layers.append(nn.Tanh())
        return nn.Sequential(*layers)
    actor_head = head(12, hidden_dims, init_func=layer_init, std=np.sqrt(2), bias_const=0.01)
    IPython.embed()




if __name__ == '__main__':
    test() # pylint: disable=no-value-for-parameter
