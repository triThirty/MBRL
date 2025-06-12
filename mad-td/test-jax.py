import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax
from flax import linen as nn
from flax.training import train_state
from collections import deque
import random

# --- Hyperparameters ---
ENV_NAME = "Walker2d-v4"
BATCH_SIZE = 256
GAMMA = 0.99
UTD = 20
MODEL_RATIO = 0.05
MODEL_UPDATE_FREQ = 1000
MAX_EPISODES = 500
MAX_STEPS = 50
LR_ACTOR = 1e-3
LR_CRITIC = 1e-3
LR_MODEL = 1e-3

# --- Replay Buffer ---
replay = deque(maxlen=200000)

# --- JAX Setup ---
key = jax.random.PRNGKey(0)


# --- Neural Networks with Flax ---
class Actor(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        x = nn.tanh(x)
        return x


class Critic(nn.Module):
    @nn.compact
    def __call__(self, state, action):
        x = jnp.concatenate([state, action], axis=-1)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x


class DynModel(nn.Module):
    obs_dim: int

    @nn.compact
    def __call__(self, state, action):
        x = jnp.concatenate([state, action], axis=-1)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(self.obs_dim + 1)(x)  # Predict next_state and reward
        next_state = x[..., :-1]
        reward = x[..., -1:]
        return next_state, reward


# --- Initialize environment ---
env = gym.make(ENV_NAME, render_mode="none")
obs, _ = env.reset()
obs_dim = obs.shape[0]
act_dim = env.action_space.shape[0]


# --- Initialize networks ---
def init_net(rng, model, sample_input, *args, **kwargs):
    if model == DynModel:
        model = model(obs_dim=obs_dim)
        params = model.init(rng, sample_input[0], sample_input[1])
    else:
        model = model(*args, **kwargs)
        params = model.init(rng, sample_input)
    return params


# Create RNG keys
key, actor_key, critic_key, model_key = jax.random.split(key, 4)

# Sample inputs
dummy_obs = jnp.ones((1, obs_dim))
dummy_act = jnp.ones((1, act_dim))

# Initialize parameters
actor_params = init_net(actor_key, Actor, dummy_obs, act_dim)
critic_params = init_net(critic_key, Critic, (dummy_obs, dummy_act))
model_params = init_net(model_key, DynModel, (dummy_obs, dummy_act))

# Target networks
target_actor_params = flax.core.frozen_dict.unfreeze(actor_params)
target_critic_params = flax.core.frozen_dict.unfreeze(critic_params)

# --- Optimizers ---
actor_optim = optax.adam(LR_ACTOR)
critic_optim = optax.adam(LR_CRITIC)
model_optim = optax.adam(LR_MODEL)

actor_opt_state = actor_optim.init(actor_params)
critic_opt_state = critic_optim.init(critic_params)
model_opt_state = model_optim.init(model_params)


# --- JIT Compiled Functions ---
@jax.jit
def actor_apply(params, obs):
    return Actor(act_dim).apply(params, obs)


@jax.jit
def critic_apply(params, obs, action):
    return Critic().apply(params, obs, action)


@jax.jit
def model_apply(params, obs, action):
    return DynModel(obs_dim).apply(params, obs, action)


@jax.jit
def update_model(model_params, model_opt_state, batch):
    s, a, r, sn = batch

    def loss_fn(params):
        pred_sn, pred_r = model_apply(params, s, a)
        loss = jnp.mean((pred_sn - sn) ** 2) + jnp.mean((pred_r - r) ** 2)
        return loss

    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(model_params)
    updates, new_opt_state = model_optim.update(grads, model_opt_state)
    new_params = optax.apply_updates(model_params, updates)
    return new_params, new_opt_state


@jax.jit
def update_critic(
    critic_params,
    critic_opt_state,
    actor_params,
    target_actor_params,
    target_critic_params,
    batch,
):
    s, a, r, sn = batch

    def loss_fn(params):
        # Target Q-value
        next_a = actor_apply(target_actor_params, sn)
        next_q = critic_apply(target_critic_params, sn, next_a)
        y = r + GAMMA * next_q

        # Current Q-value
        current_q = critic_apply(params, s, a)
        loss = jnp.mean((current_q - y) ** 2)
        return loss

    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(critic_params)
    updates, new_opt_state = critic_optim.update(grads, critic_opt_state)
    new_params = optax.apply_updates(critic_params, updates)
    return new_params, new_opt_state


@jax.jit
def update_actor(actor_params, actor_opt_state, critic_params, batch):
    s, a, r, sn = batch

    def loss_fn(params):
        new_a = actor_apply(params, s)
        q = critic_apply(critic_params, s, new_a)
        return -jnp.mean(q)

    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(actor_params)
    updates, new_opt_state = actor_optim.update(grads, actor_opt_state)
    new_params = optax.apply_updates(actor_params, updates)
    return new_params, new_opt_state


@jax.jit
def soft_update(params, target_params, tau=0.005):
    return jax.tree_map(lambda p, tp: tp * (1 - tau) + p * tau, params, target_params)


# --- Batch Sampling ---
def sample_batch():
    idx = random.sample(range(len(replay)), BATCH_SIZE)
    b = [replay[i] for i in idx]
    s = np.array([x[0] for x in b])
    a = np.array([x[1] for x in b])
    r = np.array([x[2] for x in b]).reshape(-1, 1)
    sn = np.array([x[3] for x in b])
    return s, a, r, sn


def get_mixed_batch(actor_params):
    n_model = int(BATCH_SIZE * MODEL_RATIO)
    real_s, real_a, real_r, real_sn = sample_batch()

    # Model-generated samples
    s0_idx = random.sample(range(len(replay)), n_model)
    s0 = np.array([replay[i][0] for i in s0_idx])
    a0 = np.array(actor_apply(actor_params, s0))
    sn0, r0 = model_apply(model_params, s0, a0)

    # Combine batches
    s = np.concatenate([real_s, s0], axis=0)
    a = np.concatenate([real_a, a0], axis=0)
    r = np.concatenate([real_r, r0], axis=0)
    sn = np.concatenate([real_sn, sn0], axis=0)
    return s, a, r, sn


# --- Training Loop ---
total_steps = 0
for ep in range(MAX_EPISODES):
    obs, _ = env.reset()
    ep_ret = 0
    for t in range(MAX_STEPS):
        # Select action with exploration
        s = np.array(obs).reshape(1, -1)
        a = actor_apply(actor_params, s)[0] + np.random.normal(0, 0.1, act_dim)
        a = np.clip(a, -1, 1)  # Clip to valid action range

        # Environment step
        obs2, r, done, trunc, _ = env.step(a)
        replay.append((obs, a, r, obs2))
        obs = obs2
        ep_ret += r
        total_steps += 1

        # Train dynamics model
        if total_steps % MODEL_UPDATE_FREQ == 0 and len(replay) > BATCH_SIZE:
            s_b, a_b, r_b, sn_b = sample_batch()
            batch = (
                jnp.array(s_b, dtype=jnp.float32),
                jnp.array(a_b, dtype=jnp.float32),
                jnp.array(r_b, dtype=jnp.float32),
                jnp.array(sn_b, dtype=jnp.float32),
            )
            model_params, model_opt_state = update_model(
                model_params, model_opt_state, batch
            )

        # Update policy and value networks
        if len(replay) > BATCH_SIZE:
            s_m, a_m, r_m, sn_m = get_mixed_batch(actor_params)
            batch = (
                jnp.array(s_m, dtype=jnp.float32),
                jnp.array(a_m, dtype=jnp.float32),
                jnp.array(r_m, dtype=jnp.float32),
                jnp.array(sn_m, dtype=jnp.float32),
            )

            for _ in range(UTD):
                # Critic update
                critic_params, critic_opt_state = update_critic(
                    critic_params,
                    critic_opt_state,
                    actor_params,
                    target_actor_params,
                    target_critic_params,
                    batch,
                )

                # Actor update (every 2 steps)
                if _ % 2 == 0:
                    actor_params, actor_opt_state = update_actor(
                        actor_params, actor_opt_state, critic_params, batch
                    )

                    # Update target networks
                    target_actor_params = soft_update(actor_params, target_actor_params)
                    target_critic_params = soft_update(
                        critic_params, target_critic_params
                    )

        if done or trunc:
            break

    print(f"Episode: {ep}, Return: {ep_ret:.1f}, Steps: {total_steps}")

env.close()
