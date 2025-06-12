# import gymnasium as gym
from math import e
import re
import minari
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
import argparse
from train_model import Morel
from torch.utils.tensorboard import SummaryWriter


class Maze2DDataset(Dataset):

    def __init__(self):
        # self.env = gym.make("maze2d-umaze-v1")
        # dataset = self.env.get_dataset()
        episode_dataset = minari.load_dataset("D4RL/pointmaze/umaze-v2")
        self.env = episode_dataset.recover_environment()

        # dataset = next(dataset.iterate_episodes())
        # episodes = []

        self.observations = []
        self.rewards = []
        self.actions = []
        self.truncations = []
        self.delta = []

        for episode in episode_dataset.iterate_episodes():
            self.delta.append(
                episode.observations["observation"][1:]
                - episode.observations["observation"][:-1]
            )
            self.observations.append(episode.observations["observation"][:-1])
            self.rewards.append(episode.rewards)
            self.actions.append(episode.actions)
            self.truncations.append(episode.truncations)

        self.observations = np.concatenate(self.observations)
        self.rewards = np.concatenate(self.rewards)
        self.actions = np.concatenate(self.actions)
        self.truncations = np.concatenate(self.truncations)
        self.delta = np.concatenate(self.delta)

        # Input data
        # self.source_observation = dataset.observations["observation"][:-1]
        # self.source_action = dataset.actions

        self.source_observation = self.observations
        self.source_action = self.actions

        # Output data
        # self.target_delta = self.observations[1:] - self.observations[:-1]
        # self.target_reward = self.rewards
        self.target_delta = self.delta
        self.target_reward = self.rewards

        # Normalize data
        self.delta_mean = self.target_delta.mean(axis=0)
        self.delta_std = self.target_delta.std(axis=0)

        self.reward_mean = self.target_reward.mean(axis=0)
        self.reward_std = (
            self.target_reward.std(axis=0)
            if self.target_reward.std(axis=0) != 0
            else 1.0
        )

        self.observation_mean = self.source_observation.mean(axis=0)
        self.observation_std = self.source_observation.std(axis=0)

        self.action_mean = self.source_action.mean(axis=0)
        self.action_std = self.source_action.std(axis=0)

        self.source_action = (self.source_action - self.action_mean) / self.action_std
        self.source_observation = (
            self.source_observation - self.observation_mean
        ) / self.observation_std
        self.target_delta = (self.target_delta - self.delta_mean) / self.delta_std
        self.target_reward = (self.target_reward - self.reward_mean) / self.reward_std

        # Get indices of initial states
        # self.done_indices = dataset["timeouts"][:-1]
        # self.initial_indices = np.roll(self.done_indices, 1)
        # self.initial_indices[0] = True

        # Calculate distribution parameters for initial states
        self.initial_obs = self.source_observation[self.truncations]
        self.initial_obs_mean = self.initial_obs.mean(axis=0)
        self.initial_obs_std = self.initial_obs.std(axis=0)

        # Remove transitions from terminal to initial states
        # self.source_action = np.delete(self.source_action, self.done_indices, axis=0)
        # self.source_observation = np.delete(
        #     self.source_observation, self.done_indices, axis=0
        # )
        # self.target_delta = np.delete(self.target_delta, self.done_indices, axis=0)
        # self.target_reward = np.delete(self.target_reward, self.done_indices, axis=0)

    def __getitem__(self, idx):
        feed = torch.FloatTensor(
            np.concatenate([self.source_observation[idx], self.source_action[idx]])
        )
        target = torch.FloatTensor(
            np.concatenate([self.target_delta[idx], self.target_reward[idx : idx + 1]])
        )
        return feed, target

    def __len__(self):
        return len(self.source_observation)


def main(args):
    tensorboard_writer = None
    if args.tensorboard:
        tensorboard_writer = SummaryWriter(
            log_dir=args.log_dir + args.exp_name,
            comment="MOReL",
            flush_secs=5,
        )

    # Instantiate dataset
    dynamics_data = Maze2DDataset()

    dataloader = DataLoader(dynamics_data, batch_size=128, shuffle=True)

    agent = Morel(4, 2, tensorboard_writer=tensorboard_writer)

    agent.train(dataloader, dynamics_data)

    agent.eval(dynamics_data.env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training")

    parser.add_argument("--log_dir", type=str, default="../results/")
    parser.add_argument("--tensorboard", action="store_true", default=True)
    parser.add_argument("--comet_config", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default="exp_test")
    parser.add_argument("--no_log", action="store_true")

    args = parser.parse_args()
    main(args)
