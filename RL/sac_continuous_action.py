# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import argparse
import os
import random
import time
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import stable_baselines3 as sb3
from stable_baselines3.common.buffers import ReplayBuffer
from RL.utils import set_run_name, set_track
from env.dm2gym import make_vectorized_envs

def head(in_features, hidden_dims):
    """Create a head template for actor and critic (aka. Agent network)"""
    layers = [] # 下面in_features如果是numpy.int64类型，会报错，所以要转换成int类型
    in_features = int(in_features) if isinstance(in_features, np.integer) else in_features
    for in_dim, out_dim in zip((in_features,)+hidden_dims, hidden_dims):  
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class SoftQNetwork(nn.Module):
    def __init__(self, envs, hidden_dims=(256, 256)):
        super().__init__()
        hidden_dims = (hidden_dims,) if isinstance(hidden_dims, int) else hidden_dims
        input_dim = np.array(envs.single_observation_space.shape).prod() + np.prod(envs.single_action_space.shape)
        self.Q = nn.Sequential(
                            head(input_dim, hidden_dims),
                            nn.Linear(hidden_dims[-1], 1),
                            )
    def forward(self, x, a): # Q network
        return self.Q(torch.cat([x, a], -1))


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, envs, hidden_dims=(256, 256)):
        super().__init__()
        hidden_dims = (hidden_dims,) if isinstance(hidden_dims, int) else hidden_dims
        input_dim = np.array(envs.single_observation_space.shape).prod()
        output_dim = np.array(envs.single_action_space.shape).prod()
        self.head = head(input_dim, hidden_dims)
        self.mean = nn.Linear(hidden_dims[-1], output_dim)
        self.logstd = nn.Linear(hidden_dims[-1], output_dim)
        # action rescaling
        action_low = torch.tensor(envs.single_action_space.low).view(1, -1).float()
        action_high = torch.tensor(envs.single_action_space.high).view(1, -1).float()
        self.register_buffer("low", action_low)
        self.register_buffer("high", action_high)
        self.register_buffer("action_scale", (action_high - action_low) / 2.0)
        self.register_buffer("action_bias", (action_high + action_low) / 2.0)

    def forward(self, x): #  value network
        x = self.head(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats
        return mean, log_std

    def get_action(self, x): # policy network
        mean, log_std = self(x)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


def trainer(config):
    if sb3.__version__ < "2.0":
        raise ValueError(
            "Ongoing migration: run the following command to install the new dependencies\n" + 
            "pip install \"stable_baselines3==2.0.0a1\""
        )

    ######### 0. CONFIG #########
    exp_name = config.exp_name
    track = config.track
    wandb_project_name = config.wandb_project_name
    wandb_entity = config.wandb_entity
    seed = config.seed
    torch_deterministic = config.torch_deterministic
    if config.cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif config.mps:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    env_id = config.task.env_id
    env_args = config.task
    num_envs = config.algo.num_envs
    asynchronous = config.algo.asynchronous
    hidden_dims = config.algo.hidden_dims
    total_timesteps = config.algo.total_timesteps
    q_lr = config.algo.q_lr
    policy_lr = config.algo.policy_lr
    buffer_size = config.algo.buffer_size
    batch_size = config.algo.batch_size
    gamma = config.algo.gamma
    tau = config.algo.tau
    exploration_noise = config.algo.exploration_noise
    learning_starts = config.algo.learning_starts
    policy_delay = config.algo.policy_delay
    noise_clip = config.algo.noise_clip
    alpha = config.algo.alpha
    autotune = config.algo.autotune
    save_freq = config.algo.save_freq


    ########## 1. SEED #########
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = torch_deterministic

    ########## 2. LOGGER ##########
    run_name = set_run_name(env_id, exp_name, seed, int(time.time()))
    writer = set_track(wandb_project_name, wandb_entity, run_name, config, track)

    ########## 3. ENVIRONMENT #########
    envs = make_vectorized_envs(num_envs=num_envs,
                                asynchronous=asynchronous,
                                **env_args,)
    assert isinstance(envs.single_action_space, gym.spaces.Box),\
        "only continuous action space is supported"

    ########## 4. AGENT ##########
    actor = Actor(envs, hidden_dims).to(device)
    Qnet1 = SoftQNetwork(envs, hidden_dims).to(device)
    Qnet2 = SoftQNetwork(envs, hidden_dims).to(device)
    Qnet1_target = SoftQNetwork(envs, hidden_dims).to(device)
    Qnet2_target = SoftQNetwork(envs, hidden_dims).to(device)
    Qnet1_target.load_state_dict(Qnet1.state_dict())
    Qnet2_target.load_state_dict(Qnet2.state_dict())
    Qnet_optimizer = optim.Adam(list(Qnet1.parameters()) + list(Qnet2.parameters()), lr=q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=policy_lr)

    if autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        alpha_optimizer = optim.Adam([log_alpha], lr=q_lr)
    else:
        alpha = alpha

    ########## 5. REPLAYBUFFER #########
    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=True,
    )

    ########## 6. TRAINING #########
    start_time = time.time()
    global_step = 0
    num_updates = total_timesteps // num_envs
    print(f"Start SAC...总更新次数为{num_updates}")
    obs, _ = envs.reset(seed=seed)
    for update in range(1, num_updates + 1):

        # STEP 1: get actions. If in the initial stage, take random actions, otherwise from the actor
        if global_step < learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()
        global_step += num_envs * update

        # STEP 2: execute actions in envs
        next_obs, rewards, terminateds, truncateds, infos = envs.step(actions)
        if "final_info" in infos:
            for info in infos["final_info"]:
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break

        # STEP 3: add data to replay buffer
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(truncateds):
            if d: # In our case, we never truncate the episode
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminateds, infos)
        obs = next_obs

        # STEP 4: Update, if the replay buffer is ready
        if global_step > learning_starts:
            data = rb.sample(batch_size)

            # Update two Q-networks, choose a smaller as the Q_true
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                Qnet1_next_target = Qnet1_target(data.next_observations, next_state_actions)
                Qnet2_next_target = Qnet2_target(data.next_observations, next_state_actions)
                min_Qnet_next_target = torch.min(Qnet1_next_target, Qnet2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * gamma * (min_Qnet_next_target).view(-1)

            Qnet1_a_values = Qnet1(data.observations, data.actions).view(-1)
            Qnet2_a_values = Qnet2(data.observations, data.actions).view(-1)
            Qnet1_loss = F.mse_loss(Qnet1_a_values, next_q_value)
            Qnet2_loss = F.mse_loss(Qnet2_a_values, next_q_value)
            Qnet_loss = Qnet1_loss + Qnet2_loss

            Qnet_optimizer.zero_grad()
            Qnet_loss.backward()
            Qnet_optimizer.step()

            if update % policy_delay == 0:  # Delayed update support
                for _ in range(policy_delay):
                    # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    Qnet1_pi = Qnet1(data.observations, pi)
                    Qnet2_pi = Qnet2(data.observations, pi)
                    min_Qnet_pi = torch.min(Qnet1_pi, Qnet2_pi).view(-1)
                    actor_loss = ((alpha * log_pi) - min_Qnet_pi).mean()
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha * (log_pi + target_entropy)).mean()

                        alpha_optimizer.zero_grad()
                        alpha_loss.backward()
                        alpha_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if update % policy_delay == 0:
                for param, target_param in zip(Qnet1.parameters(), Qnet1_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for param, target_param in zip(Qnet2.parameters(), Qnet2_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            if update % 8 == 0:
                writer.add_scalar("losses/Qnet1_values", Qnet1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/Qnet2_values", Qnet2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/Qnet1_loss", Qnet1_loss.item(), global_step)
                writer.add_scalar("losses/Qnet2_loss", Qnet2_loss.item(), global_step)
                writer.add_scalar("losses/Qnet_loss", Qnet_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                SPS = int(global_step / (time.time() - start_time))
                TPU = float((time.time() - start_time) / global_step / 60)
                RT  = float((num_updates - update) * TPU)
                print(f"Update={update}, SPS={SPS}, TPU={TPU:.2f}min, RT={RT/60:.2f}h", end="\r")
                writer.add_scalar("charts/SPS", SPS, global_step)
                writer.add_scalar("charts/RestTime", RT/60, global_step)
                writer.add_scalar("charts/TimePerUpdate", TPU, global_step)
                if autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

            # Checkpoints
            if update % save_freq == 0:
                torch.save(actor, f"checkpoints/{run_name}-update{update}.pth")
                for filename in os.listdir("checkpoints"):
                    if filename == f"{run_name}-update{update-save_freq}.pth":
                        os.remove(f"checkpoints/{filename}")
    # Final save
    torch.save(actor, f"checkpoints/{run_name}-update{update}.pth")
    envs.close()
    writer.close()