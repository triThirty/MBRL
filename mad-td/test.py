import time
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np

from torch.utils.tensorboard import SummaryWriter


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# --- Hyperparameters ---
ENV_NAME = "Walker2d-v4"
BATCH_SIZE = 256
GAMMA = 0.99
UTD = 20
MODEL_RATIO = 0.05
MODEL_UPDATE_FREQ = 1000
MAX_EPISODES = 10000
MAX_STEPS = 50
LR_ACTOR = 1e-3
LR_CRITIC = 1e-3
LR_MODEL = 1e-3
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


# --- Util Set Up ---
writer = SummaryWriter(log_dir=f"runs/{ENV_NAME}_{SEED}_{time.strftime('%m%d-%H%M%S')}")

# --- Replay Buffer ---
replay = deque(maxlen=200000)


# --- Actor & Critic ---
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(), nn.Linear(256, act_dim), nn.Tanh()
        )

    def forward(self, s):
        return self.net(s)


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256), nn.ReLU(), nn.Linear(256, 1)
        )

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))


# --- Dynamics Model ---
class DynModel(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256), nn.ReLU(), nn.Linear(256, obs_dim + 1)
        )  # predict next_obs and reward

    def forward(self, s, a):
        out = self.net(torch.cat([s, a], -1))
        return out[..., :-1], out[..., -1:]


# --- Initialize environment and networks ---
env = gym.make(ENV_NAME, render_mode="none")
obs, _ = env.reset()
obs_dim = obs.shape[0]
act_dim = env.action_space.shape[0]

actor = Actor(obs_dim, act_dim).to(DEVICE)
critic = Critic(obs_dim, act_dim).to(DEVICE)
actor_t = Actor(obs_dim, act_dim).to(DEVICE)
critic_t = Critic(obs_dim, act_dim).to(DEVICE)
actor_t.load_state_dict(actor.state_dict())
critic_t.load_state_dict(critic.state_dict())

dyn_model = DynModel(obs_dim, act_dim).to(DEVICE)

opt_actor = optim.Adam(actor.parameters(), lr=LR_ACTOR)
opt_critic = optim.Adam(critic.parameters(), lr=LR_CRITIC)
opt_model = optim.Adam(dyn_model.parameters(), lr=LR_MODEL)


# --- Soft update ---
def soft_update(net, net_t, tau=0.005):
    for p, p_t in zip(net.parameters(), net_t.parameters()):
        p_t.data.mul_(1 - tau)
        p_t.data.add_(p.data * tau)


# --- Sample batch and mixing model data ---
def sample_batch():
    idx = random.sample(range(len(replay)), BATCH_SIZE)
    b = [replay[i] for i in idx]
    s = torch.FloatTensor([x[0] for x in b]).to(DEVICE)
    a = torch.FloatTensor([x[1] for x in b]).to(DEVICE)
    r = torch.FloatTensor([x[2] for x in b]).unsqueeze(1).to(DEVICE)
    sn = torch.FloatTensor([x[3] for x in b]).to(DEVICE)
    return s, a, r, sn


def get_mixed_batch():
    n_model = int(BATCH_SIZE * MODEL_RATIO)
    real_s, real_a, real_r, real_sn = sample_batch()
    # Sample model-generated
    s0_idx = random.sample(range(len(replay)), n_model)
    s0 = torch.FloatTensor([replay[i][0] for i in s0_idx]).to(DEVICE)
    with torch.no_grad():
        a0 = actor(s0)
        sn0, r0 = dyn_model(s0, a0)
    sn0_clamped = sn0  # no env clipping here
    s = torch.cat([real_s, s0], dim=0)
    a = torch.cat([real_a, a0], dim=0)
    r = torch.cat([real_r, r0], dim=0)
    sn = torch.cat([real_sn, sn0_clamped], dim=0)
    return s, a, r, sn


# --- Training Loop ---
total_steps = 0
for ep in range(MAX_EPISODES):
    obs, _ = env.reset()
    ep_ret = 0
    for t in range(MAX_STEPS):
        s = torch.FloatTensor(obs).to(DEVICE)
        with torch.no_grad():
            a = actor(s).cpu().numpy() + np.random.normal(0, 0.1, act_dim)
        obs2, r, done, trunc, _ = env.step(a)
        replay.append((obs, a, r, obs2))
        obs = obs2
        ep_ret += r
        total_steps += 1

        # Train dynamics model periodically
        if total_steps % MODEL_UPDATE_FREQ == 0 and len(replay) > BATCH_SIZE:
            ss, aa, rr, snn = sample_batch()
            pred_sn, pred_r = dyn_model(ss, aa)
            loss_model = ((pred_sn - snn) ** 2 + (pred_r - rr) ** 2).mean()
            opt_model.zero_grad()
            loss_model.backward()
            opt_model.step()

        # Perform updates UTD times
        if len(replay) > BATCH_SIZE:
            for _ in range(UTD):
                s_b, a_b, r_b, sn_b = get_mixed_batch()
                with torch.no_grad():
                    at = actor_t(sn_b)
                    qt = critic_t(sn_b, at)
                    y = r_b + GAMMA * qt
                loss_q = ((critic(s_b, a_b) - y) ** 2).mean()
                opt_critic.zero_grad()
                loss_q.backward()
                opt_critic.step()
                if _ % 2 == 0:  # actor update freq 2
                    loss_pi = -critic(s_b, actor(s_b)).mean()
                    opt_actor.zero_grad()
                    loss_pi.backward()
                    opt_actor.step()
                    soft_update(actor, actor_t)
                    soft_update(critic, critic_t)

        if done or trunc:
            break

    writer.add_scalar("Episode/Return", ep_ret, ep)
    print(ep, "Return:", ep_ret)

env.close()
writer.close()
