from ast import Pass
import copy
from json import tool
import torch
import torch.nn.functional as F
from torch.optim import Adam
from rl_sac.utils import soft_update, hard_update
from rl_sac.model import GaussianPolicy, Critic, PointCritic, PointGaussianPolicy
from imitation.utils import img_to_tensor
import numpy as np

class SAC(object):
    def __init__(self, args, obs_shape, action_dim, max_action):
        self.agent_type = 'Img' if not args.use_pcl else 'Point'
        self.args = args
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu")

        # self.critic = QNetwork(obs_shape, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic = critics[self.agent_type](args, obs_shape, action_dim, args.hidden_dim).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = critics[self.agent_type](args, obs_shape, action_dim, args.hidden_dim).to(self.device)
        hard_update(self.critic_target, self.critic)

        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        if self.automatic_entropy_tuning is True:
            real_action_dim = np.sum(self.args.action_mask).astype(np.int)
            self.target_entropy = -torch.Tensor(real_action_dim).to(self.device)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=args.lr)
        action_mask = torch.FloatTensor(self.args.action_mask).to(self.device).reshape(1, -1)
        self.policy = policies[self.agent_type](args, obs_shape, action_dim, max_action, action_mask, args.hidden_dim).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
    def select_action_pcl(self, dough_pcl, dough_pcl_len, tool_pcl, tool_pcl_len, goal_pcl, goal_pcl_len, evaluate=False):
        with torch.no_grad():
            obs = np.concatenate([np.array(dough_pcl), np.array(tool_pcl)],axis=1)
            obs_len = np.concatenate([np.array(dough_pcl_len).reshape(self.args.num_env, 1), np.array(tool_pcl_len).reshape(self.args.num_env, 1)], axis=1)
            goal_obs = np.array(goal_pcl)
            goal_len = np.array(goal_pcl_len).reshape(self.args.num_env, 1)

            obs = torch.FloatTensor(obs).to(self.device)
            obs_len = torch.tensor(obs_len).to(self.device)
            goal_obs = torch.FloatTensor(goal_obs).to(self.device)
            goal_len = torch.tensor(goal_len).to(self.device)
            if evaluate is False:
                action, _, _ = self.policy.sample(obs, obs_len, goal_obs, goal_len)
            else:
                _, _, action = self.policy.sample(obs, obs_len, goal_obs, goal_len)
            del obs, obs_len, goal_obs, goal_len
        return action.cpu().data.numpy().reshape(self.args.num_env, -1)

    def select_action(self, obs, goal_obs, evaluate=False):
        if isinstance(obs, list):  # Batch env
            with torch.no_grad():
                obs = (np.array(obs) if self.args.use_pcl else img_to_tensor(np.array(obs), mode=self.args.img_mode)).to(self.device)
                goal_obs = (np.array(goal_obs) if self.args.use_pcl else img_to_tensor(np.array(goal_obs), mode=self.args.img_mode)).to(self.device)
                if evaluate is False:
                    action, _, _ = self.policy.sample(obs, goal_obs)
                else:
                    _, _, action = self.policy.sample(obs, goal_obs)
                return action.cpu().data.numpy().reshape(self.args.num_env, -1)
        else:
            with torch.no_grad():
                obs = (obs[None] if self.args.use_pcl else img_to_tensor(obs[None], mode=self.args.img_mode)).to(self.device)
                goal_obs = (goal_obs[None] if self.args.use_pcl else img_to_tensor(goal_obs[None], mode=self.args.img_mode)).to(self.device)
                if evaluate is False:
                    action, _, _ = self.policy.sample(obs, goal_obs)
                else:
                    _, _, action = self.policy.sample(obs, goal_obs)
                return action.cpu().data.numpy().flatten()

    def update_parameters(self, replay_buffer, batch_size):
        # Sample a batch from memory
        if self.args.use_pcl == 'partial_pcl':
            obs, obs_len, goal_obs, goal_len, n_obs, n_obs_len, action, reward, not_done = replay_buffer.her_sample(batch_size)
            obs = torch.FloatTensor(obs).to(self.device)
            obs_len = torch.tensor(obs_len).to(self.device)
            goal_obs = torch.FloatTensor(goal_obs).to(self.device)
            goal_len = torch.tensor(goal_len).to(self.device)
            n_obs = torch.FloatTensor(n_obs).to(self.device)
            n_obs_len = torch.tensor(n_obs_len).to(self.device)
        else:
            obs, goal_obs, action, next_obs, reward, not_done = replay_buffer.her_sample(batch_size)
            obs = img_to_tensor(obs, mode=self.args.img_mode).to(self.device, non_blocking=True)
            goal_obs = img_to_tensor(goal_obs, mode=self.args.img_mode).to(self.device, non_blocking=True)
            next_obs = img_to_tensor(next_obs, mode=self.args.img_mode).to(self.device, non_blocking=True)
        action = torch.FloatTensor(action).to(self.device, non_blocking=True)
        reward = torch.FloatTensor(reward).to(self.device, non_blocking=True)
        not_done = torch.FloatTensor(not_done).to(self.device, non_blocking=True)
        with torch.no_grad():
            if self.args.use_pcl == 'partial_pcl':
                next_obs_action, next_obs_log_pi, _ = self.policy.sample(n_obs, n_obs_len, goal_obs, goal_len)
                qf1_next_target, qf2_next_target = self.critic_target(n_obs, n_obs_len, goal_obs, goal_len, next_obs_action)
            else:
                next_obs_action, next_obs_log_pi, _ = self.policy.sample(next_obs, goal_obs)
                qf1_next_target, qf2_next_target = self.critic_target(next_obs, goal_obs, next_obs_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_obs_log_pi
            next_q_value = reward + not_done * self.gamma * (min_qf_next_target)
        if self.args.use_pcl == 'partial_pcl':
            qf1, qf2 = self.critic(obs, obs_len, goal_obs, goal_len, action)
        else:   
            qf1, qf2 = self.critic(obs, goal_obs, action)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        if self.args.use_pcl == 'partial_pcl':
            pi, log_pi, _ = self.policy.sample(obs, obs_len, goal_obs, goal_len)
            qf1_pi, qf2_pi = self.critic(obs, obs_len, goal_obs, goal_len, pi)
        else:
            pi, log_pi, _ = self.policy.sample(obs, goal_obs)
            qf1_pi, qf2_pi = self.critic(obs, goal_obs, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        soft_update(self.critic_target, self.critic, self.tau)
        del obs, n_obs, goal_obs
        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic.pth")
        # torch.save(self.critic_optim.state_dict(), filename + "_critic_optimizer.pth")
        torch.save(self.policy.state_dict(), filename + "_actor.pth")
        # torch.save(self.policy_optim.state_dict(), filename + "_actor_optimizer.pth")

    def load(self, filename, evaluate=True):
        self.critic.load_state_dict(torch.load(filename + "_critic.pth"))
        self.critic_target = copy.deepcopy(self.critic)

        self.policy.load_state_dict(torch.load(filename + "_actor.pth"))
        if evaluate:
            self.policy.eval()
            self.critic.eval()
            self.critic_target.eval()
        else:
            self.policy.train()
            self.critic.train()
            self.critic_target.train()
        print('actor loaded from ', filename + "_actor.pth")

critics = {
    'Img': Critic,
    'Point': PointCritic
}
policies = {
    'Img': GaussianPolicy,
    'Point': PointGaussianPolicy
}