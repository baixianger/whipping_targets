"""
DDPG (Deep Deterministic Policy Gradient) for continuous action space, with some tricks:
    1. Policy network computes an deterministic action but not sampled from a distribution.
    2. Polyak averaging of the target network.
    3. Time-correlated OU noise for exploration.
    4. Delayed update of the policy network (not in the original DDPG).
"""
import os
import random
import time
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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


class QNetwork(nn.Module):
    def __init__(self, envs, hidden_dims=(256, 256)):
        super().__init__()
        hidden_dims = (hidden_dims,) if isinstance(hidden_dims, int) else hidden_dims
        input_dim = np.array(envs.single_observation_space.shape).prod() + np.prod(envs.single_action_space.shape)
        self.Q = nn.Sequential(
                            head(input_dim, hidden_dims),
                            nn.Linear(hidden_dims[-1], 1),
                            )
    def forward(self, x, a):
        return self.Q(torch.cat([x, a], -1))


class Actor(nn.Module):
    def __init__(self, envs, hidden_dims=(256, 256)):
        super().__init__()
        hidden_dims = (hidden_dims,) if isinstance(hidden_dims, int) else hidden_dims
        input_dim = np.array(envs.single_observation_space.shape).prod()
        output_dim = np.array(envs.single_action_space.shape).prod()
        self.actor = nn.Sequential(
                                head(input_dim, hidden_dims),
                                nn.Linear(hidden_dims[-1], output_dim),
                                nn.Tanh(),
                                )
        # action rescaling
        action_low = torch.tensor(envs.single_action_space.low).view(1, -1).float()
        action_high = torch.tensor(envs.single_action_space.high).view(1, -1).float()
        self.register_buffer("low", action_low)
        self.register_buffer("high", action_high)
        self.register_buffer("action_scale", (action_high - action_low) / 2.0)
        self.register_buffer("action_bias", (action_high + action_low) / 2.0)

    def forward(self, x):
        return self.actor(x) * self.action_scale + self.action_bias


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
    wandb_group = config.algo.name
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
    num_envs = int(config.algo.num_envs)
    asynchronous = config.algo.asynchronous
    hidden_dims = config.algo.hidden_dims
    num_updates = int(config.algo.num_updates)
    total_timesteps = int(config.algo.total_timesteps)
    learning_rate = config.algo.learning_rate
    buffer_size = int(config.algo.buffer_size)
    batch_size = int(config.algo.batch_size)
    gamma = config.algo.gamma
    tau = config.algo.tau
    anneal_noise = config.algo.anneal_noise
    noise_dacay = config.algo.noise_dacay
    exploration_noise = config.algo.exploration_noise
    learning_starts = int(config.algo.learning_starts)
    policy_delay = int(config.algo.policy_delay)
    noise_clip = config.algo.noise_clip
    save_freq = int(config.algo.save_freq)

    ########## 1. SEED #########
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = torch_deterministic

    ########## 2. LOGGER ##########
    run_name = set_run_name(env_id, exp_name, seed, int(time.time()))
    writer = set_track(wandb_project_name, wandb_entity, wandb_group, run_name, config, track)

    ########## 3. ENVIRONMENT #########
    envs = make_vectorized_envs(num_envs=num_envs,
                                asynchronous=asynchronous,
                                **env_args,)
    assert isinstance(envs.single_action_space, gym.spaces.Box),\
        "only continuous action space is supported"

    ########## 4. AGENT ##########
    actor = Actor(envs, hidden_dims).to(device)
    Qnet = QNetwork(envs, hidden_dims).to(device)
    actor_target = Actor(envs, hidden_dims).to(device)
    Qnet_target = QNetwork(envs, hidden_dims).to(device)
    actor_target.load_state_dict(actor.state_dict())
    Qnet_target.load_state_dict(Qnet.state_dict())
    Qnet_optimizer = optim.Adam(list(Qnet.parameters()), lr=learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=learning_rate)

    ########## 5. REPLAYBUFFER #########
    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        num_envs,
        handle_timeout_termination=False,
    )

    ########## 6. TRAINING #########
    start_time = time.time()
    global_step = 0
    num_updates = num_updates if num_updates else total_timesteps // num_envs
    print(f"Start DDPG...总更新次数为{num_updates}")
    obs, _ = envs.reset(seed=seed)
    for update in range(1, num_updates + 1):
        
        # STEP 1: get actions. If in the initial stage, take random actions, otherwise from the actor
        if global_step < learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            print(f"ReplayBuffer Warmup [{global_step:<5d}/{learning_starts:d}]", end="\r")
        else:
            with torch.no_grad():
                # TODO: decrease the exploration noise during training
                if anneal_noise:
                    frac = 1.0 - update / num_updates
                    sampling_noise = frac * exploration_noise
                else:
                    sampling_noise = exploration_noise
                actions = actor(torch.Tensor(obs).to(device))
                actions += torch.normal(0, actor.action_scale * sampling_noise)
                actions = torch.clamp(actions, actor.low, actor.high).cpu().numpy()
        global_step += num_envs * update

        # STEP 2: execute actions in envs
        next_obs, rewards, terminateds, truncateds, infos = envs.step(actions)
        if "final_info" in infos:
            for info in infos["final_info"]:
                episodic_return = info["episode"]["r"][0]
                # print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break

        # STEP 3: add data to replay buffer
        #         向量化环境后, 会自动重设, 所以当一个回合结束后返回的状态是对应环境重设后的初始状态
        #         环境真正的最后一个状态保存在infos["final_observation"]中
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(terminateds):
            if d: # In our case, we never truncate the episode
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminateds, infos)
        # import IPython; IPython.embed()
        obs = next_obs

        # STEP 4: Update, if the replay buffer is ready
        if global_step > learning_starts:
            data = rb.sample(batch_size)

            # Update Q-Network, minimize the TD-error, delta_Q = Q(s,a) - (r + gamma * Q(s',a'))
            with torch.no_grad():
                next_state_actions = actor_target(data.next_observations)
                Q_value_next = Qnet_target(data.next_observations, next_state_actions)
                Q_value_true = data.rewards.flatten() + (1 - data.dones.flatten()) * gamma * (Q_value_next).view(-1)

            Q_value_pred = Qnet(data.observations, data.actions).view(-1)
            Qnet_loss = F.mse_loss(Q_value_pred, Q_value_true)
            Qnet_optimizer.zero_grad()
            Qnet_loss.backward()
            Qnet_optimizer.step()

            # Update Policy network, maximize Q(s,a)
            if update % policy_delay == 0: # DDPG 原始算法不会延迟更新actor
                actor_loss = -Qnet(data.observations, actor(data.observations)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for param, target_param in zip(Qnet.parameters(), Qnet_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            if update % 8 == 0: # every 2048 samples
                writer.add_scalar("losses/Qnet_loss", Qnet_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/Qnet_values", Q_value_pred.mean().item(), global_step)
                SPS = int(global_step / (time.time() - start_time))
                TPU = float((time.time() - start_time) / update / 60)
                RT  = float((num_updates - update) * TPU / 60)
                print(f"Update={update}, SPS={SPS}, TPU={TPU:.2f}min, RT={RT:.2f}h", end="\r")
                writer.add_scalar("charts/SPS", SPS, global_step)
                writer.add_scalar("charts/RestTime", RT, global_step)
                writer.add_scalar("charts/TimePerUpdate", TPU, global_step)


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