import torch
from torch._C import dtype
import torch.nn as nn
import numpy as np
from imitation import utils
from torch.optim import Adam
import torch.nn.functional as F
from plb.models.cnn_encoder import CNNEncoder
from plb.models.pointnet_encoder import PointNetEncoder, PointNetEncoder2, PointNetEncoder3


def softclipping(x, l, r):
    x = r - F.softplus(r - x)
    x = l + F.softplus(x - l)
    x = torch.clamp(x, max=r)
    return x



class ImgActor(nn.Module):
    """ Image based actor"""

    def __init__(self, args, obs_shape, action_dim, hidden_dim=1024):
        super().__init__()
        self.args = args
        obs_shape = (obs_shape[0], obs_shape[1], obs_shape[2] + obs_shape[2] * args.frame_stack)
        self.encoder = CNNEncoder(obs_shape, args.actor_feature_dim)
        self.mlp = nn.Sequential(nn.Linear(args.actor_feature_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, action_dim))
        self.done_mlp = nn.Sequential(nn.Linear(args.actor_feature_dim, hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, 1))

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, detach_encoder=False):
        obs = self.encoder(obs, detach=detach_encoder)

        action = self.mlp(obs)
        done = self.done_mlp(obs)

        self.outputs['mu'] = action
        return action, done

    def log(self, logger, step):
        raise NotImplementedError
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_actor/{k}_hist', v, step)

        for i, m in enumerate(self.trunk):
            if type(m) == nn.Linear:
                logger.log_param(f'train_actor/fc{i}', m, step)

class PointActor2(nn.Module):
    """ PointCloud based actor"""

    def __init__(self, args, obs_shape, action_dim, hidden_dim=1024):
        super().__init__()
        self.args = args
        self.encoders = nn.ModuleList([PointNetEncoder2(obs_shape[-1]) for _ in range(3)])
        self.mlp = nn.Sequential(nn.Linear(256*3, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, action_dim))
        self.done_mlp = nn.Sequential(nn.Linear(256*3, 512),
                                      nn.ReLU(),
                                      nn.Linear(512, 256),
                                      nn.ReLU(),
                                      nn.Linear(256, 1))

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, data, detach_encoder=False):

        s_dough, s_tool, s_tar = self.encoders[0](data['dough'], detach=detach_encoder), \
                                    self.encoders[1](data['tool'], detach=detach_encoder), \
                                    self.encoders[2](data['goal'], detach=detach_encoder)
        s = torch.cat([s_dough, s_tool, s_tar], dim=1)
        action = self.mlp(s)
        done = self.done_mlp(s)
        # hardcode for good initialization
        # TODO: divide by 20 
        action = action / 5.
        done = done / 5.

        # self.outputs['mu'] = action
        return action, done

    def log(self, logger, step):
        raise NotImplementedError
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_actor/{k}_hist', v, step)

        for i, m in enumerate(self.trunk):
            if type(m) == nn.Linear:
                logger.log_param(f'train_actor/fc{i}', m, step)

class PointActor(nn.Module):
    """ PointCloud based actor"""

    def __init__(self, args, obs_shape, action_dim, hidden_dim=1024):
        super().__init__()
        self.args = args
        self.encoder = PointNetEncoder(obs_shape[-1])
        self.mlp = nn.Sequential(nn.Linear(1024, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, action_dim))
        self.done_mlp = nn.Sequential(nn.Linear(1024, 512),
                                      nn.ReLU(),
                                      nn.Linear(512, 256),
                                      nn.ReLU(),
                                      nn.Linear(256, 1))

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, data, detach_encoder=False):
        obs = self.encoder(data, detach=detach_encoder)
        action = self.mlp(obs)
        done = self.done_mlp(obs)
        # hardcode for good initialization
        # import pdb; pdb.set_trace()
        # TODO: divide by 20 
        action = action / 5.
        done = done / 5.

        # self.outputs['mu'] = action
        return action, done

    def log(self, logger, step):
        raise NotImplementedError
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_actor/{k}_hist', v, step)

        for i, m in enumerate(self.trunk):
            if type(m) == nn.Linear:
                logger.log_param(f'train_actor/fc{i}', m, step)

class PointActorToolParticle(nn.Module):
    """ PointCloud based actor"""

    def __init__(self, args, obs_shape, action_dim, hidden_dim=1024):
        super().__init__()
        self.args = args
        self.encoder = PointNetEncoder3(obs_shape[-1])

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, data, detach_encoder=False):
        x = self.encoder(data, detach=detach_encoder)
        return x

class Agent(object):
    def __init__(self, args, solver, obs_shape, action_dim, num_tools, device='cuda'):
        """obs_shape should be [img_size x img_size x img_channel] or [num_points x 3]"""
        self.args = args
        self.solver = solver
        print('class Agent: obs_shape: ', obs_shape)
        self.obs_shape = obs_shape
        self.num_tools = num_tools
        self.actor_type = args.actor_type
        self.actors = nn.ModuleList([actors[self.actor_type](args, obs_shape, action_dim).to(device) for _ in range(num_tools)])
        self.optim = Adam(list(self.actors.parameters()), lr=args.il_lr)
        self.criterion = torch.nn.MSELoss()
        self.terminate_early = False
        self.device = device
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, verbose=True, factor=0.5, patience=100)
        self.max_pts = {
            'tool_pcl': 100 if self.args.gt_tool else 200,
            'dough_pcl': 1000,
            'goal_pcl': 1000
        }
    def act(self, obs, goal, tid, stats=None):
        """ Batch inference! """
        if isinstance(self.actors[tid], ImgActor):
            obs = torch.cat([obs, goal], dim=1)  # concat the channel for image, concat the batch for point cloud
            action, done = self.actors[tid](obs)
        elif isinstance(self.actors[tid], PointActor):
            obs = torch.cat([obs, goal], dim=1)  # concat the channel for image, concat the batch for point cloud
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
            action, done = self.actors[tid](data)
        elif isinstance(self.actors[tid], PointActor2):
            data = {}
            pos, feature = torch.split(obs, [3, self.obs_shape[-1] - 3], dim=-1)
            goal, goal_feat = torch.split(goal, [3, self.obs_shape[-1] - 3], dim=-1)
            if feature.shape[-1] == 0:
                dough_feat = None
                tool_feat = None
                goal_feat = None
            else:
                dough_feat = feature[:, :1000, :].reshape(-1, feature.shape[-1])
                tool_feat = feature[:, 1000:1100, :].reshape(-1, feature.shape[-1])
                goal_feat = goal_feat.reshape(-1, goal_feat.shape[-1])
            n = obs.shape[0]
            dough_pos = pos[:, :1000, :]
            tool_pos = pos[:, 1000:1100, :]
            data['dough'] = (dough_feat, dough_pos.reshape(-1, 3), torch.arange(n).repeat_interleave(dough_pos.shape[1]).to(obs.device, non_blocking=True))
            data['tool'] = (tool_feat, tool_pos.reshape(-1, 3), torch.arange(n).repeat_interleave(tool_pos.shape[1]).to(obs.device, non_blocking=True))
            data['goal'] = (goal_feat, goal.reshape(-1, 3), torch.arange(n).repeat_interleave(goal.shape[1]).to(obs.device, non_blocking=True))
            action, done = self.actors[tid](data)
        return action, done



    def act_partial_pcl(self, obs, obs_len, goal, goal_len, tid, eval=False, tool_xyz=None):
        # assert isinstance(self.actors[tid], PointActor), "Only implemented for PN++"
        data = {}
        n = obs.shape[0]
        obs = torch.cat([obs, goal], dim=1)  # concat the channel for image, concat the batch for point cloud
        obs_len = torch.cat([obs_len, goal_len], dim=1) # Bx3
        # dough: [0,0,1] tool: 0,1,0, target: 1,0,0
        sums = torch.sum(obs_len, dim=1)
        batch = torch.repeat_interleave(torch.arange(obs.shape[0]).to(self.device), sums)
        onehot = torch.cat([
            torch.FloatTensor([0,0,1]).repeat(self.max_pts['dough_pcl'], 1), 
            torch.FloatTensor([0,1,0]).repeat(self.max_pts['tool_pcl'], 1),
            torch.FloatTensor([1,0,0]).repeat(self.max_pts['goal_pcl'], 1),
                ], dim=0).to(self.device)
        onehot = onehot.repeat((obs.shape[0], 1, 1))
        x = onehot.reshape(-1, 3)
        pos = obs.reshape(-1, 3)
        points_idx = pos.sum(dim=1) != 0
        x = x[points_idx]
        pos = pos[points_idx]
        # remove noise during test
        if not eval and 'obs_noise' in self.args.__dict__ and self.args.obs_noise > 0:
            pos = (torch.rand(pos.shape, device=pos.device) - 0.5) * 2 * self.args.obs_noise + pos
        if self.args.frame == 'tool':
            tool_xyz = torch.repeat_interleave(tool_xyz, sums, dim=0).reshape(-1, 3)
            pos -= tool_xyz

        data['pos'] = pos
        data['x'] = x
        data['batch'] = batch
        # print(data['pos'].shape, data['x'].shape, data['batch'].shape)

        if self.actor_type == 'PointActorToolParticle':
            tool_onehot = torch.FloatTensor([0,1,0]).to(self.device)
            mask = torch.all(torch.eq(tool_onehot, x), dim=1)
            pred_fl = self.actors[tid](data)
            return pred_fl[mask].reshape(obs.shape[0], -1, 3)

        else:
            action, done = self.actors[tid](data)
            return action, done

    def train_tool(self, data_batch, agent_ids=None, stats=None, pcl=''):
        self.actors[0].train()
        obses, goals, tool_flow = \
            data_batch['obses'], data_batch['goal_obses'], data_batch['tool_flow']
        flow_losses = []
        tids = range(1)
        if agent_ids is None:
            agent_ids = range(1)
        for tid, agent_id  in zip(tids, agent_ids):
            if 'obs_noise' in self.args.__dict__ and self.args.obs_noise > 0 and pcl != 'partial_pcl':
                obses[tid] = (torch.rand(obses[tid].shape, device=obses[tid].device) - 0.5) * 2 * self.args.obs_noise + obses[tid]
                goals[tid] = (torch.rand(goals[tid].shape, device=obses[tid].device) - 0.5) * 2 * self.args.obs_noise + goals[tid]
            if pcl == 'partial_pcl':
                ## TODO: remove the if-else case to merge full pcl and partial pcl dataloading
                if self.args.frame == 'tool':
                    tool_xyz = data_batch['tool_xyz'][tid]
                else:
                    tool_xyz = None
                pred_tool_flow = self.act_partial_pcl(obses[tid], data_batch['obses_pcl_len'][tid], goals[tid], \
                                                                data_batch['goals_pcl_len'][tid], agent_id, tool_xyz=tool_xyz)
            else:    
                raise NotImplementedError
            flow_loss = self.criterion(pred_tool_flow, tool_flow[tid])
            flow_losses.append(flow_loss)
            # print("gt tool flow:", tool_flow[tid][0, :10])
            # breakpoint()
        l = sum(flow_losses)

        self.optim.zero_grad()
        l.backward()
        self.optim.step()
        # self.scheduler.step(l)
        return {'avg_flow_loss': (sum(flow_losses) / len(flow_losses)).item(),
                'lr': self.scheduler.optimizer.param_groups[0]['lr']}


    def train(self, data_batch, agent_ids=None, stats=None, pcl=''):
        self.actors[0].train()
        obses, goals, dones, actions = \
            data_batch['obses'], data_batch['goal_obses'], data_batch['dones'], data_batch['actions']
        action_losses, done_losses = [], []
        tids = range(1)
        if agent_ids is None:
            agent_ids = range(1)
        for tid, agent_id  in zip(tids, agent_ids):
            if 'obs_noise' in self.args.__dict__ and self.args.obs_noise > 0 and pcl != 'partial_pcl':
                obses[tid] = (torch.rand(obses[tid].shape, device=obses[tid].device) - 0.5) * 2 * self.args.obs_noise + obses[tid]
                goals[tid] = (torch.rand(goals[tid].shape, device=obses[tid].device) - 0.5) * 2 * self.args.obs_noise + goals[tid]
            if pcl == 'partial_pcl':
                ## TODO: remove the if-else case to merge full pcl and partial pcl dataloading
                if self.args.frame == 'tool':
                    tool_xyz = data_batch['tool_xyz'][tid]
                else:
                    tool_xyz = None
                pred_actions, pred_dones = self.act_partial_pcl(obses[tid], data_batch['obses_pcl_len'][tid], goals[tid], \
                                                                data_batch['goals_pcl_len'][tid], agent_id, tool_xyz=tool_xyz)
            else:    
                pred_actions, pred_dones = self.act(obses[tid], goals[tid], agent_id)
            action_loss = self.criterion(pred_actions, actions[tid])
            done_loss = self.criterion(pred_dones, dones[tid][:, None])

            action_losses.append(action_loss)
            done_losses.append(done_loss)
    
        l = sum(action_losses) + sum(done_losses)

        self.optim.zero_grad()
        l.backward()
        self.optim.step()
        # self.scheduler.step(l)
        return {'avg_action_loss': (sum(action_losses) / len(action_losses)).item(),
                'avg_done_loss': (sum(done_losses) / len(done_losses)).item(),
                'lr': self.scheduler.optimizer.param_groups[0]['lr']}

    def eval(self, data_batch, tid=0, pcl=''):
        self.actors[0].eval()
        with torch.no_grad():
            obses, goals, dones, actions = \
                data_batch['obses'], data_batch['goal_obses'], data_batch['dones'], data_batch['actions']
            if pcl == 'partial_pcl':
                ## TODO: remove the if-else case to merge full pcl and partial pcl dataloading
                if self.args.frame == 'tool':
                    tool_xyz = data_batch['tool_xyz'][tid]
                else:
                    tool_xyz = None
                pred_actions, pred_dones = self.act_partial_pcl(obses[tid], data_batch['obses_pcl_len'][tid], goals[tid], data_batch['goals_pcl_len'][tid], tid, eval=True, tool_xyz=tool_xyz)
            else:    
                pred_actions, pred_dones = self.act(obses[tid], goals[tid], tid)
            action_loss = self.criterion(pred_actions, actions[tid]).item()
            done_loss = self.criterion(pred_dones, dones[tid][:, None]).item()
            return {'eval_action_loss': action_loss, 'eval_done_loss': done_loss}

    def get_argmax_points(self):
        assert isinstance(self.actors[0], PointActor2)
        points = []
        for encoder in self.actors[0].encoders:
            point = encoder.sa3_module.argmax_points.detach().cpu().numpy()
            points.append(point)
        return points

    def save(self, path):
        torch.save({'actors': self.actors.state_dict()}, path)

    def load(self, path):
        ckpt = torch.load(path)
        self.actors.load_state_dict(ckpt['actors'])
        print('Agent loaded from ' + path)


actors = {"Img": ImgActor, \
        "Point": PointActor, \
        "Point2": PointActor2, \
        "PointActorToolParticle": PointActorToolParticle}