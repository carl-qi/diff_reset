import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from imitation.agent import Encoder
from plb.models.cnn_encoder import CNNEncoder
from plb.models.pointnet_encoder import PointNetEncoder, PointNetEncoder2

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


# Initialize weights
def weight_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

class PointCritic(nn.Module):
    def __init__(self, args, obs_shape, action_dim, hidden_dim=256):
        super().__init__()
        args.feature_dim = 1024
        self.args = args
        self.max_pts = {
            'tool_pcl': 100 if self.args and self.args.gt_tool else 200,
            'dough_pcl': 1000,
            'goal_pcl': 1000
        }
        self.encoder = PointNetEncoder(obs_shape[-1])
        
        self.critic1 = nn.Sequential(nn.Linear(args.feature_dim + action_dim, hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(hidden_dim, hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(hidden_dim, 1))
        self.critic2 = nn.Sequential(nn.Linear(args.feature_dim + action_dim, hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(hidden_dim, hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(hidden_dim, 1))

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, obs_len, goal, goal_len, action, detach_encoder=False):
        data = {}
        n = obs.shape[0]
        obs = torch.cat([obs, goal], dim=1)  # concat the channel for image, concat the batch for point cloud
        obs_len = torch.cat([obs_len, goal_len], dim=1) # Bx3
        # dough: [0,0,1] tool: 0,1,0, target: 1,0,0
        sums = torch.sum(obs_len, dim=1)
        batch = torch.repeat_interleave(torch.arange(obs.shape[0]).to(obs.device), sums)
        onehot = torch.cat([
            torch.FloatTensor([0,0,1]).repeat(self.max_pts['dough_pcl'], 1), 
            torch.FloatTensor([0,1,0]).repeat(self.max_pts['tool_pcl'], 1),
            torch.FloatTensor([1,0,0]).repeat(self.max_pts['goal_pcl'], 1),
                ], dim=0).to(obs.device)
        onehot = onehot.repeat((obs.shape[0], 1, 1))
        x = onehot.reshape(-1, 3)
        pos = obs.reshape(-1, 3)
        points_idx = pos.sum(dim=1) != 0
        x = x[points_idx]
        pos = pos[points_idx]

        data['pos'] = pos
        data['x'] = x
        data['batch'] = batch

        obs = self.encoder(data, detach=detach_encoder)

        oa = torch.cat([obs, action], dim=1)

        q1 = self.critic1(oa)
        q2 = self.critic2(oa)
        return q1, q2

class PointGaussianPolicy(nn.Module):
    def __init__(self, args, obs_shape, action_dim, max_action, action_mask, hidden_dim=256):
        super().__init__()
        args.feature_dim = 1024
        self.args = args
        self.encoder = PointNetEncoder(obs_shape[-1])
        
        self.mlp = nn.Sequential(nn.Linear(args.feature_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.ReLU())

        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

        self.apply(weight_init)

        # action rescaling - TODO Check the actual action scale
        self.action_scale = torch.tensor(1.)
        self.action_bias = torch.tensor(0.)
        self.action_mask = action_mask
        self.max_pts = {
            'tool_pcl': 100 if self.args and self.args.gt_tool else 200,
            'dough_pcl': 1000,
            'goal_pcl': 1000
        }

    def forward(self, obs, obs_len, goal, goal_len, detach_encoder=False):
        data = {}
        n = obs.shape[0]
        obs = torch.cat([obs, goal], dim=1)  # concat the channel for image, concat the batch for point cloud
        obs_len = torch.cat([obs_len, goal_len], dim=1) # Bx3
        # dough: [0,0,1] tool: 0,1,0, target: 1,0,0
        sums = torch.sum(obs_len, dim=1)
        batch = torch.repeat_interleave(torch.arange(obs.shape[0]).to(obs.device), sums)
        onehot = torch.cat([
            torch.FloatTensor([0,0,1]).repeat(self.max_pts['dough_pcl'], 1), 
            torch.FloatTensor([0,1,0]).repeat(self.max_pts['tool_pcl'], 1),
            torch.FloatTensor([1,0,0]).repeat(self.max_pts['goal_pcl'], 1),
                ], dim=0).to(obs.device)
        onehot = onehot.repeat((obs.shape[0], 1, 1))
        x = onehot.reshape(-1, 3)
        pos = obs.reshape(-1, 3)
        points_idx = pos.sum(dim=1) != 0
        x = x[points_idx]
        pos = pos[points_idx]

        data['pos'] = pos
        data['x'] = x
        data['batch'] = batch
        obs = self.encoder(data, detach=detach_encoder)
        feat = self.mlp(obs)
        mean = self.mean_linear(feat)
        log_std = self.log_std_linear(feat)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, obs, obs_len, goal_obs, goal_obs_len):
        mean, log_std = self.forward(obs, obs_len, goal_obs, goal_obs_len)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob * self.action_mask
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        action = action * self.action_mask
        mean = mean * self.action_mask
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(PointGaussianPolicy, self).to(device)

class Critic(nn.Module):
    def __init__(self, args, obs_shape, action_dim, hidden_dim=256):
        super().__init__()
        self.args = args
        obs_shape = (obs_shape[0], obs_shape[1], obs_shape[2] * 2)  # Goal conditioned
        self.encoder = Encoder(obs_shape, args.feature_dim)
        self.critic1 = nn.Sequential(nn.Linear(args.feature_dim + action_dim, hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(hidden_dim, hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(hidden_dim, 1))
        self.critic2 = nn.Sequential(nn.Linear(args.feature_dim + action_dim, hidden_dim),
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


class GaussianPolicy(nn.Module):
    def __init__(self, args, obs_shape, action_dim, max_action, action_mask, hidden_dim=256):
        super(GaussianPolicy, self).__init__()

        self.args = args
        obs_shape = (obs_shape[0], obs_shape[1], obs_shape[2] * 2)  # Goal conditioned
        self.encoder = Encoder(obs_shape, args.feature_dim)
        self.mlp = nn.Sequential(nn.Linear(args.feature_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.ReLU())

        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

        self.apply(weight_init)

        # action rescaling - TODO Check the actual action scale
        self.action_scale = torch.tensor(1.)
        self.action_bias = torch.tensor(0.)
        self.action_mask = action_mask

    def forward(self, obs, goal_obs, detach_encoder=False):
        obs = torch.cat([obs, goal_obs], dim=1)
        obs = self.encoder(obs, detach=detach_encoder)
        feat = self.mlp(obs)
        mean = self.mean_linear(feat)
        log_std = self.log_std_linear(feat)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, obs, goal_obs):
        mean, log_std = self.forward(obs, goal_obs)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob * self.action_mask
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        action = action * self.action_mask
        mean = mean * self.action_mask
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)
