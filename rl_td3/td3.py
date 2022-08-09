import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from imitation.agent import Encoder
from imitation.utils import weight_init, img_to_tensor

device = 'cuda'


# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
    def __init__(self, args, obs_shape, action_dim, max_action, hidden_dim=1024):
        super().__init__()
        self.args = args
        obs_shape = (obs_shape[0], obs_shape[1], obs_shape[2] * 2)  # Goal conditioned
        self.encoder = Encoder(obs_shape, args.feature_dim)
        self.mlp = nn.Sequential(nn.Linear(args.feature_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, action_dim))
        self.max_action = max_action
        self.action_mask = torch.FloatTensor(self.args.action_mask).to(device).reshape(1, -1)
        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, goal_obs, detach_encoder=False):
        obs = torch.cat([obs, goal_obs], dim=1)
        obs = self.encoder(obs, detach=detach_encoder)
        action = self.mlp(obs)
        action = self.max_action * torch.tanh(action)
        action *= self.action_mask
        self.outputs['mu'] = action
        return action


class Critic(nn.Module):
    def __init__(self, args, obs_shape, action_dim, hidden_dim=1024):
        super().__init__()
        self.args = args
        obs_shape = (obs_shape[0], obs_shape[1], obs_shape[2] * 2)  # Goal conditioned
        self.encoder = Encoder(obs_shape, args.feature_dim)
        self.critic1 = nn.Sequential(nn.Linear(args.feature_dim + action_dim, hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(hidden_dim, hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(hidden_dim, hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(hidden_dim, 1))
        self.critic2 = nn.Sequential(nn.Linear(args.feature_dim + action_dim, hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(hidden_dim, hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(hidden_dim, hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(hidden_dim, 1))

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, goal_obs, action, detach_encoder=False):
        obs = torch.cat([obs, goal_obs], dim=1)
        obs = self.encoder(obs, detach=detach_encoder)

        oa = torch.cat([obs, action], 1)

        q1 = self.critic1(oa)
        q2 = self.critic2(oa)
        return q1, q2

    def Q1(self, obs, goal_obs, action, detach_encoder=False):
        obs = torch.cat([obs, goal_obs], dim=1)
        obs = self.encoder(obs, detach=detach_encoder)

        oa = torch.cat([obs, action], 1)

        q1 = self.critic1(oa)
        return q1


class TD3(object):
    def __init__(
      self,
      args,
      obs_shape,
      action_dim,
      max_action,
      discount=0.99,
      tau=0.005,
      policy_noise=0.2,
      noise_clip=0.5,
      policy_freq=2
    ):

        self.actor = Actor(args, obs_shape, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.lr)

        self.critic = Critic(args, obs_shape, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.lr)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0
        self.args = args

    def select_action(self, obs, goal_obs):
        if isinstance(obs, list):  # Batch env
            with torch.no_grad():
                obs = img_to_tensor(np.array(obs), mode=self.args.img_mode).to(device)
                goal_obs = img_to_tensor(np.array(goal_obs), mode=self.args.img_mode).to(device)
                return self.actor(obs, goal_obs).cpu().data.numpy().reshape(self.args.num_env, -1)
        else:
            with torch.no_grad():
                obs = img_to_tensor(obs[None], mode=self.args.img_mode).to(device)
                goal_obs = img_to_tensor(goal_obs[None], mode=self.args.img_mode).to(device)
                return self.actor(obs, goal_obs).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=128):
        self.total_it += 1

        # Sample replay buffer
        obs, goal_obs, action, next_obs, reward, not_done = replay_buffer.her_sample(batch_size)
        obs = img_to_tensor(obs, mode=self.args.img_mode).to(device, non_blocking=True)
        goal_obs = img_to_tensor(goal_obs, mode=self.args.img_mode).to(device, non_blocking=True)
        next_obs = img_to_tensor(next_obs, mode=self.args.img_mode).to(device, non_blocking=True)
        action = torch.FloatTensor(action).to(device, non_blocking=True)
        reward = torch.FloatTensor(reward).to(device, non_blocking=True)
        not_done = torch.FloatTensor(not_done).to(device, non_blocking=True)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
              torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
              self.actor_target(next_obs, goal_obs) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_obs, goal_obs, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, goal_obs, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(obs, goal_obs, self.actor(obs, goal_obs)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic.pth")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer.pth")

        torch.save(self.actor.state_dict(), filename + "_actor.pth")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer.pth")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic.pth"))
        # self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer.pth"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor.pth"))
        # self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer.pth"))
        self.actor_target = copy.deepcopy(self.actor)

