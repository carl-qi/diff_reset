import torch
import torch.nn as nn
import numpy as np
from imitation import utils
from torch.optim import Adam
import torch.nn.functional as F
from plb.models.cnn_encoder import CNNEncoder
from plb.models.pointnet_encoder import PointNetEncoder, PointNetEncoder2

class PointEBM(nn.Module):
    """ PointCloud based ebm"""

    def __init__(self, args, obs_shape, action_dim, hidden_dim=1024):
        super().__init__()
        self.args = args
        self.encoder = PointNetEncoder(obs_shape[-1])
        self.mlp = nn.Sequential(nn.Linear(1024+6, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 1))
        # self.done_mlp = nn.Sequential(nn.Linear(1024, 512),
                                    #   nn.ReLU(),
                                    #   nn.Linear(512, 256),
                                    #   nn.ReLU(),
                                    #   nn.Linear(256, 1))

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, data, detach_encoder=False):
        obs = self.encoder(data, detach=detach_encoder)
        # breakpoint()
        action = data['actions']
        obs_tiled = obs.repeat_interleave(action.shape[0] // obs.shape[0], dim=0)
        score = self.mlp(torch.cat([obs_tiled,action], dim=1)) # B * n+1
        score = score.reshape(obs.shape[0], -1)
        # B = obs.shape[0]
        # score = torch.cat([
            # self.mlp(torch.cat([obs, action[i*B:(i+1)*B]], dim=1)) 
            # for i in range(action.shape[0]//B)])
        return score

    def log(self, logger, step):
        raise NotImplementedError
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_actor/{k}_hist', v, step)

        for i, m in enumerate(self.trunk):
            if type(m) == nn.Linear:
                logger.log_param(f'train_actor/fc{i}', m, step)






class Agent(object):
    def __init__(self, args, solver, obs_shape, action_dim, num_tools, device='cuda'):
        """obs_shape should be [img_size x img_size x img_channel] or [num_points x 3]"""
        self.args = args
        self._num_counter_examples = args.n_counter_example
        self.temp = args.temp
        self.sigma_init = 0.01
        self.n_iters = 3
        self.n_inference_samples = 30000
        self.K = 0.5
        self.solver = solver
        self.terminate_early = False
        print('class Agent: obs_shape: ', obs_shape)
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.num_tools = num_tools
        self.actor_type = args.actor_type
        self.actors = nn.ModuleList([actors[self.actor_type](args, obs_shape, action_dim).to(device) for _ in range(num_tools)])
        self.optim = Adam(list(self.actors.parameters()), lr=args.il_lr)
        self.device = device
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim, gamma=0.99)
        self.scheduled_step = 0
        self.debug = {}


    def act(self, obs, goal, tid, stats={}):
        """ Not Batch inference! """
        assert stats != {}
        with torch.no_grad():
            samples = torch.rand(self.n_inference_samples, self.action_dim, device=self.device) \
                        * (stats['y_max']-stats['y_min']) + stats['y_min'] # n_infr_sampl x action_dim
            mean = torch.zeros(self.action_dim).to(self.device)
            sigma = self.sigma_init * torch.ones(self.action_dim).to(self.device)
            for i in range(self.n_iters):
                # breakpoint()
                data = self.organize_data(obs, goal, samples, tid, stats, generate_counter=False)
                scores = self.actors[tid](data) # 1 x n_infr_sampl
                self.debug['energy'] = torch.max(-scores)
                probs = F.softmax(-scores / self.temp, dim=-1)
                # breakpoint()
                if i < self.n_iters-1:
                    idxes = torch.multinomial(probs, self.n_inference_samples, replacement=True).squeeze(0)
                    samples = samples[idxes]
                    # samples = torch.clamp(samples + torch.normal(mean, sigma), \
                        # stats['y_min'], stats['y_max']) # torch 1.5 on seuss is stupid
                    
                    samples = samples + torch.normal(mean, sigma)
                    samples = torch.where(samples > stats['y_max'], stats['y_max'], samples)
                    samples = torch.where(samples < stats['y_min'], stats['y_min'], samples)
                    sigma = self.K * sigma
            action, done = samples[torch.argmax(probs)], torch.zeros(1).to(self.device)
            # print("probs_mean:", torch.mean(probs), "probs_max", torch.max(probs))
            # print('action:', action)
            return [action], [done]

    def organize_data(self, obs, goal, actions, tid, stats, generate_counter=True):
        if isinstance(self.actors[tid], PointEBM):
            if generate_counter:
                counter_samples = torch.rand(actions.shape[0], self._num_counter_examples, actions.shape[1], device=self.device) \
                    * (stats['y_max']-stats['y_min']) + stats['y_min'] # B x n x action_dim
                positive_samples = torch.unsqueeze(actions, 1) # B x 1 x action_dim
                full_actions = torch.cat([counter_samples, positive_samples], dim=1)  # B x (n+1) x action_dim
                full_actions = full_actions.reshape(-1, self.action_dim) # B*(n+1) x action_dim
            else:
                full_actions = actions
            obs = torch.cat([obs, goal], dim=1)  # concat the channel for image, concat the batch for point cloud
            obs, full_actions = self.normalize(obs, full_actions, stats)
            # Preprocess obs into the shape that encoder requires
            data = {}
            # TODO: without hard coded dimensions
            pos, feature = torch.split(obs, [3, self.obs_shape[-1] - 3], dim=-1)
            # dough: [0,0,1] tool: 0,1,0, target: 1,0,0
            n = obs.shape[0]
            onehot = torch.zeros((obs.shape[0], obs.shape[1], 3)).to(obs.device)
            onehot[:, :1000, 2:3] += 1
            onehot[:, 1000:1100, 1:2] += 1
            onehot[:, 1100:, 0:1] += 1
            x = torch.cat([feature, onehot], dim=-1)
            data['x'] = x.reshape(-1, 3+feature.shape[-1])
            data['pos'] = pos.reshape(-1, 3)
            data['batch'] = torch.arange(n).repeat_interleave(obs.shape[1]).to(obs.device, non_blocking=True)
            data['actions'] = full_actions
            return data
        else:
            raise NotImplementedError

    def train(self, data_batch, stats={}, agent_ids=None):
        self.actors[0].train()
        assert stats != {}
        obses, goals, dones, actions = \
        data_batch['obses'], data_batch['goal_obses'], data_batch['dones'], data_batch['actions']
        infonce_losses = []
        tids = range(1)
        if agent_ids is None:
            agent_ids = range(1)
        for tid, agent_id  in zip(tids, agent_ids):
            if 'obs_noise' in self.args.__dict__ and self.args.obs_noise > 0:
                obses[tid] = (torch.rand(obses[tid].shape, device=obses[tid].device) - 0.5) * 2 * self.args.obs_noise + obses[tid]
                goals[tid] = (torch.rand(goals[tid].shape, device=obses[tid].device) - 0.5) * 2 * self.args.obs_noise + goals[tid]

            data = self.organize_data(obses[tid], goals[tid], actions[tid], tid, stats)
            # data['counter_samples'] = counter_samples
            score = self.actors[tid](data) # B x n+1
            loss = self.info_nce_loss(score, temp=self.temp)

            infonce_losses.append(loss)
        l = sum(infonce_losses)
        self.optim.zero_grad()
        l.backward()
        self.optim.step()
        self.scheduled_step += 1
        # if self.scheduled_step % 100 == 0:
            # self.scheduler.step()
        return {'avg_infonce_loss': (sum(infonce_losses) / len(infonce_losses)).item(),
                'lr': self.scheduler.optimizer.param_groups[0]['lr']}

                

    def diagnostic(self, data_batch, stats, tid=0):
        self.actors[0].eval()
        obses, goals, dones, actions = \
        data_batch['obses'], data_batch['goal_obses'], data_batch['dones'], data_batch['actions']
        # import pdb; pdb.set_trace()
        data_expert = self.organize_data(obses[tid], goals[tid], actions[tid], tid, stats, generate_counter=False)
        energies_exp = -self.actors[tid](data_expert).item() # 1 x n_infr_sampl
        self.act(obses[tid], goals[tid], tid, stats=stats)
        energies_policy = self.debug['energy'].item()
        # print("expert_energy:", energies_exp, "policy_energy:", energies_policy)
        return energies_exp, energies_policy
                

    def eval(self, data_batch, tid=0):
        return {}

    def save(self, path):
        torch.save({'actors': self.actors.state_dict()}, path)

    def load(self, path):
        ckpt = torch.load(path)
        self.actors.load_state_dict(ckpt['actors'])
        print('Agent loaded from ' + path)

    def normalize(self, obs, actions, stats):
        b, n, d = obs.shape
        obs = (obs.reshape(-1, d) - stats['state_mean']) / stats['state_std']

        actions = (actions - stats['action_mean']) / stats['action_std']
        return obs.reshape(b, n, d), actions

    def info_nce_loss(self, scores, temp=1.0):
        loss_fn = nn.KLDivLoss(reduction='batchmean') # mean(sum(l, axis=-1), axis=0)
        softmaxed_predictions = F.softmax(-scores / temp, dim=-1)
        labels = torch.zeros_like(softmaxed_predictions).to(self.device) # [B x n+1] with 1 in column [:, -1]
        labels[:, -1] += 1
        per_example_loss = loss_fn(softmaxed_predictions, labels)

        return per_example_loss


actors = {
    'Point': PointEBM,
}