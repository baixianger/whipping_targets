"""Versatile PPO algorithm for continuous action space"""
# pylint: disable=unused-import
# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes
# pylint: disable=line-too-long
# pylint: disable=invalid-name
import os
import random
import time
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.distributions.normal import Normal
from torch.distributions.beta import Beta
import gymnasium as gym
from RL.utils import set_run_name, set_track
from env.dm2gym import make_vectorized_envs



def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Trailed orthogonal weight initialization.
    For the policy network: initialization of weights with scaling 1.414 and 0.01.
    For the value network: initialization of weights with scaling 1.414 and 1.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def head(in_features, hidden_dims, init_func=layer_init, **kwargs):
    """Create a head template for actor and critic (aka. Agent network)"""
    layers = []
    for in_dim, out_dim in zip((in_features,)+hidden_dims, hidden_dims):
        layers.append(init_func(nn.Linear(in_dim, out_dim)), **kwargs)
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)

class Agent(nn.Module):
    """This is a separate MLP networks for policy and value network.

    Value Network:
        Input: observation
        Output: value
    
    Policy Network:
        Description: Action's distribution is either normal or beta distribution and is independent or not
        Input: observation, action
        Output: action, logprob, entropy
    """
    def __init__(self, envs, hidden_dims=(32, 64),
                 action_dist="normal", independent=True,):
        super().__init__()
        hidden_dims = (hidden_dims,) if isinstance(hidden_dims, int) else hidden_dims

        # VALUE NETWORK
        self.critic = nn.Sequential(
            head(np.array(envs.single_observation_space.shape).prod(), hidden_dims, layer_init),
            layer_init(nn.Linear(hidden_dims[-1], 1), std=1.0),
        )

        # POLICY NETWORK
        self.action_low = torch.tensor(envs.single_action_space.low).float().to(next(self.parameters()).device)
        self.action_high = torch.tensor(envs.single_action_space.high).float().to(next(self.parameters()).device)
        if action_dist == "normal":
            self.actor_mean = nn.Sequential(
                head(np.array(envs.single_observation_space.shape).prod(), hidden_dims, layer_init),
                layer_init(nn.Linear(hidden_dims[-1], np.prod(envs.single_action_space.shape)), std=0.01), nn.Tanh()
            )
            if independent:
                self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))
            else:
                self.actor_logstd = nn.Sequential(
                    head(np.array(envs.single_observation_space.shape).prod(), hidden_dims, layer_init),
                    layer_init(nn.Linear(hidden_dims[-1], np.prod(envs.single_action_space.shape)), std=0.01)
                )
            self._policy = self._get_normal_action
        elif action_dist == "beta":
            self.activation = nn.Softmax()
            self.actor_alpha = nn.Sequential(
                head(np.array(envs.single_observation_space.shape).prod(), hidden_dims, layer_init),
                layer_init(nn.Linear(hidden_dims[-1], np.prod(envs.single_action_space.shape)), std=0.01)
            )
            self.actor_beta = nn.Sequential(
                head(np.array(envs.single_observation_space.shape).prod(), hidden_dims, layer_init),
                layer_init(nn.Linear(hidden_dims[-1], np.prod(envs.single_action_space.shape)), std=0.01)
            )
            self._policy = self._get_beta_action
        else:
            raise ValueError(f"Unsupported action distribution: {action_dist}")

    def forward(self, x, action=None):
        """Get action, logprob, entropy and value from the observation."""
        return *self.policy(x, action), self.critic(x)

    def policy(self, x, action=None):
        """Get action, logprob, entropy from the policy network. 
        If action is None, sample an action from the learned distribution.
        If action is give, get the probability under the learned distribution."""
        return self._policy(x, action)

    def value(self, x):
        """Get value from the critic network."""
        return self.critic(x)

    def _get_beta_action(self, x, action=None):
        action_alpha = (self.activation(self.actor_alpha(x)) + 1)
        action_beta = (self.activation(self.actor_beta(x)) + 1)
        probs = Beta(action_alpha, action_beta)
        if action is None:
            _action = probs.sample()
            log_prob = probs.log_prob(_action).sum(1, keepdim=True)
            action = self._rescale_action(_action)
        else:
            _action = self._scale_action(action)
            log_prob = probs.log_prob(_action).sum(1, keepdim=True)
        return action, log_prob, probs.entropy().sum(1, keepdim=True)

    def _get_normal_action(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        log_prob = probs.log_prob(action).sum(1, keepdim=True)
        return action, log_prob, probs.entropy().sum(1, keepdim=True)

    def _scale_action(self, action):
        """Scale action from [low, high] to [0, 1]."""
        return (action - self.action_low) / (self.action_high - self.action_low)

    def _rescale_action(self, action):
        """Rescale action from [0, 1] to [low, high]."""
        return action * (self.action_high - self.action_low) + self.action_low

class Buffer:
    """Buffer for offline RL
    Features:
        Recurrent sampling from the environment: instead of start with in initial observation everytime.
        GAE: Generalized Advantage Estimation
        DataLoader: feed data to pytorch model
    """
    def __init__(self, agent, envs, buffer_size, buffer_batch, batch_size, shuffle=True, seed=None):
        assert buffer_size % buffer_batch == 0,\
            "buffer_size must be divisible by buffer_batch"
        self.agent = agent
        self.envs = envs
        self.seed = seed
        self.device = next(agent.parameters()).device
        self.indices = np.arange(buffer_size)
        self.buffer_size = buffer_size
        self.buffer_batch = buffer_batch
        self.buffer_steps = buffer_size // buffer_batch
        self.batch_size = batch_size
        self.suffle = shuffle
        self.obs_dim = envs.single_observation_space.shape[0]
        self.action_dim = envs.single_action_space.shape[0]
        self.flatten_dim = self.obs_dim + self.action_dim + 6
        self.next_obs = None
        self.next_done = None
        self.global_step = 0
        self.data = None
        self.is_flatten = False

    def __iter__(self):
        self.view(-1)
        self.is_flatten = True
        if self.suffle is True:
            self.indices = torch.randperm(self.buffer_size)
        else:
            self.indices = torch.arange(self.buffer_size)
        return self

    def __next__(self):
        if self.indices.size(0) == 0:
            raise StopIteration
        if self.indices.size(0) < self.batch_size:
            batch_indices = self.indices
            self.indices = torch.tensor([])
        else:
            batch_indices = self.indices[:self.batch_size]
            self.indices = self.indices[self.batch_size:]
        return (
            self.data["obs"][batch_indices],
            self.data["actions"][batch_indices],
            self.data["logprobs"][batch_indices],
            self.data["rewards"][batch_indices],
            self.data["dones"][batch_indices],
            self.data["values"][batch_indices],
            self.data["advantages"][batch_indices],
            self.data["returns"][batch_indices],
            )

    def reset_data(self):
        self.data = {
            "obs": torch.zeros((self.buffer_steps, self.buffer_batch, self.obs_dim)).to(self.device),
            "actions": torch.zeros((self.buffer_steps, self.buffer_batch, self.action_dim)).to(self.device),
            "logprobs": torch.zeros((self.buffer_steps, self.buffer_batch, 1)).to(self.device),
            "rewards": torch.zeros((self.buffer_steps, self.buffer_batch, 1)).to(self.device),
            "dones": torch.zeros((self.buffer_steps, self.buffer_batch, 1)).to(self.device),
            "values": torch.zeros((self.buffer_steps, self.buffer_batch, 1)).to(self.device),
            "advantages": torch.zeros((self.buffer_steps, self.buffer_batch, 1)).to(self.device),
            "returns": torch.zeros((self.buffer_steps, self.buffer_batch, 1)).to(self.device),
            }

    def view(self, *size):
        """Change data shape in order to compatiable to training and sampling."""
        self.data["obs"] = self.data["obs"].view(*size, self.obs_dim)
        self.data["actions"] = self.data["actions"].view(*size, self.action_dim)
        self.data["logprobs"] = self.data["logprobs"].view(*size, 1)
        self.data["rewards"] = self.data["rewards"].view(*size, 1)
        self.data["dones"] = self.data["dones"].view(*size, 1)
        self.data["values"] = self.data["values"].view(*size, 1)
        self.data["advantages"] = self.data["advantages"].view(*size, 1)
        self.data["returns"] = self.data["returns"].view(*size, 1)

    def sampling(self, agent:Agent, envs:gym.vector.VectorEnv, writer): # pylint: disable=too-many-locals
        """Sampling trajectories from the environment."""  
        if self.data is None:
            self.reset_data()
            # 第一次采样时, 需要reset环境，作为初始observation
            next_obs, _ = envs.reset(seed=self.seed)
            self.next_obs = torch.Tensor(next_obs).to(self.device)
            self.next_done = torch.zeros((self.buffer_batch, 1)).to(self.device)
            self.global_step = 0
        else:
            self.view(self.buffer_steps, self.buffer_batch)
            self.is_flatten = False

        for step in range(0, self.buffer_steps): # batch = steps * envs
            self.global_step += 1 * self.buffer_batch  # 总采样次数
            self.data["obs"][step] = self.next_obs     # (buffer_batch, obs_dim)
            self.data["dones"][step] = self.next_done  # (buffer_batch, 1)
            with torch.no_grad():
                action, logprob, _, value = agent(self.next_obs)
                self.data["values"][step] = value      # (buffer_batch, 1)
                self.data["actions"][step] = action    # (buffer_batch, action_dim)
                self.data["logprobs"][step] = logprob  # (buffer_batch, 1)
                next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
                done = np.logical_or(terminated, truncated)
                self.data["rewards"][step, :, 0] = torch.Tensor(reward).to(self.device) # (buffer_batch,)
                self.next_obs = torch.Tensor(next_obs).to(self.device)
                self.next_done = torch.Tensor(done).view(-1, 1).to(self.device)

            # Only print when at least 1 env is done
            if "final_info" not in infos:
                continue
            for i, info in enumerate(infos["final_info"]):
                # Skip the envs that are not done
                if info is None:
                    continue
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], self.global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], self.global_step)
        # obs = self.data["obs"][step][-1][-3:]
        # episodic_return = self.data["rewards"].mean().item() * 2
        # print(f"\tGlobal_step={self.global_step}, Episodic_return={episodic_return}, Obs={obs}")

    def GAE(self, agent:Agent, gamma, gae_lambda):
        """Get Generalized Advantage Estimation and Flatten the data into a unified tensor."""
        assert self.is_flatten is False, "Data must NOT be flattened before GAE."
        with torch.no_grad():
            next_value = agent.value(self.next_obs)
            lastgaelam = 0
            for t in reversed(range(self.buffer_steps)):
                if t == self.buffer_steps - 1: # 如果最后一步完成了, v_next = 0, 否则v_next = v(s_T)
                    nextnonterminal = 1.0 - self.next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.data["dones"][t + 1]
                    nextvalues = self.data["values"][t + 1]
                delta = self.data["rewards"][t] + gamma * nextvalues * nextnonterminal - self.data["values"][t]
                lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
                self.data["advantages"][t] = lastgaelam
            self.data["returns"] = self.data["advantages"] + self.data["values"]

# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
# pylint: disable=too-many-branches
def trainer(config):
    """Train PPO algorithm with offline RL."""
    exp_name = config.exp_name
    env_id = config.task.env_id
    ppo_args = config.algo         # alogrithm related arguments
    env_args = config.task         # task related arguments
    device = torch.device("cpu")
    if config.cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif config.mps:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # 1.SEEDING
    seed = config.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = config.torch_deterministic

    # 2.PROFILING wandb setting
    track = config.track
    wandb_project_name = config.wandb_project_name
    wandb_entity = config.wandb_entity
    run_name = set_run_name(env_id, exp_name, seed, int(time.time()))
    writer = set_track(wandb_project_name, wandb_entity, run_name, config, track)

    # 3.ENVIRONMENT
    envs = make_vectorized_envs(**env_args,
                                num_envs=ppo_args.num_envs,
                                asynchronous=ppo_args.asynchronous,
                                gamma=ppo_args.gamma)
    assert isinstance(envs.single_action_space, gym.spaces.Box),\
        "only continuous action space is supported"

    # 4.AGENT INITIALIZATION
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=ppo_args.learning_rate, eps=1e-5)

    # 6.REPLAY BUFFER
    buffer = Buffer(agent, envs, ppo_args.buffer_size, ppo_args.num_envs, ppo_args.batch_size, seed)

    # 7.LEARNING LOOP
    start_time = time.time()
    num_updates = ppo_args.total_timesteps // ppo_args.batch_size
    print(f"Start PPO...总更新次数为{num_updates}")
    for update in range(1, num_updates + 1):

        # learning rate decay from learning_rate to 0
        if ppo_args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * ppo_args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # STEP 1: Sampling
        buffer.sampling(agent, envs, writer)
        buffer.GAE(agent, ppo_args.gamma, ppo_args.gae_lambda)

        # STEP 2: Training policy and value network
        #         每次重采样后, 迭代update_epochs次, 默认10
        #         设计迭代次数是update_epochs * (buffer_size / batch_size)
        clipfracs = []

        for i in range(ppo_args.update_epochs):
            for obs, actions, logprobs, _, _, values, advantages, returns in buffer:
                _, newlogprob, entropy, newvalue = agent(obs, actions) # pylint: disable=not-callable
                logratio = newlogprob - logprobs
                ratio = logratio.exp() # pi(a|s) / pi_old(a|s) 重要性采样
                with torch.no_grad(): # 记录KL散度超过阈值的次数
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > ppo_args.clip_coef).float().mean().item()]

                if ppo_args.norm_adv: # 在minibatch范围内归一化advantage
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -advantages * ratio
                pg_loss2 = -advantages * torch.clamp(ratio, 1 - ppo_args.clip_coef, 1 + ppo_args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if ppo_args.clip_vloss:
                    v_loss_unclipped = (newvalue - returns) ** 2
                    v_clipped = values + torch.clamp(newvalue - values,
                                                     -ppo_args.clip_coef, ppo_args.clip_coef,)
                    v_loss_clipped = (v_clipped - returns) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - returns) ** 2).mean()

                entropy_loss = entropy.mean() # Maximum高斯的熵，多探索
                loss = pg_loss - ppo_args.ent_coef * entropy_loss + v_loss * ppo_args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), ppo_args.max_grad_norm)
                optimizer.step()

            if ppo_args.target_kl is not None:
                if approx_kl > ppo_args.target_kl:
                    break

        values = buffer.data["values"].view(-1)
        returns = buffer.data["returns"].view(-1)
        y_pred, y_true = values.cpu().numpy(), returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        global_step = buffer.global_step
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        SPS = int(global_step / (time.time() - start_time))
        TPU = (time.time() - start_time) / update
        RT  = (num_updates - update) * TPU
        print(f"Update={update}, SPS={SPS}, TPU={TPU/60:.2f}min, RT={RT/60:.2f}min", end="\r")
        writer.add_scalar("charts/SPS", SPS, global_step)

        # Checkpoints
        freq = 50
        if update % freq == 0:
            torch.save(agent, f"checkpoints/{run_name}-update{update}.pth")
            # delete old checkpoints
            for filename in os.listdir("checkpoints"):
                if filename == f"{run_name}-update{update-freq}.pth":
                    os.remove(f"checkpoints/{filename}")
    # Final save
    torch.save(agent, f"checkpoints/{run_name}-update{update}.pth")
    envs.close()
    writer.close()
