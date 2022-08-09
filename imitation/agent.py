import torch
import torch.nn as nn
import numpy as np
from imitation import utils
from torch.optim import Adam
from imitation.encoder.vae import VAE
import torch.nn.functional as F


def softclipping(x, l, r):
    x = r - F.softplus(r - x)
    x = l + F.softplus(x - l)
    x = torch.clamp(x, max=r)
    return x


class Encoder(nn.Module):
    """Convolutional encoder for image-based observations."""

    def __init__(self, obs_shape, feature_dim):
        super().__init__()

        assert len(obs_shape) == 3
        self.num_layers = 4
        self.num_filters = 32
        self.output_dim = 35
        self.output_logits = False
        self.feature_dim = feature_dim

        self.convs = nn.ModuleList([
            nn.Conv2d(obs_shape[-1], self.num_filters, 3, stride=2),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1)
        ])

        self.head = nn.Sequential(
            nn.Linear(20000, self.feature_dim),
            nn.LayerNorm(self.feature_dim))

        self.outputs = dict()

    def forward_conv(self, obs):
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv
        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)
        if detach:
            h = h.detach()

        out = self.head(h)
        if not self.output_logits:
            out = torch.tanh(out)

        self.outputs['out'] = out

        return out

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_encoder/{k}_hist', v, step)
            if len(v.shape) > 2:
                logger.log_image(f'train_encoder/{k}_img', v[0], step)

        for i in range(self.num_layers):
            logger.log_param(f'train_encoder/conv{i + 1}', self.convs[i], step)


class Actor(nn.Module):
    """ Image based actor"""

    def __init__(self, args, obs_shape, action_dim, hidden_dim=1024):
        super().__init__()
        self.args = args
        self.encoder = Encoder(obs_shape, args.actor_feature_dim)
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


class FeasibilityPredictor(nn.Module):
    """ Predict success and score"""

    def __init__(self, args, z_dim, hidden_dim=1024):
        """ bin_succ: if true, use binary classification and otherwise use regression"""
        super().__init__()
        self.args = args
        self.bin_succ = args.bin_succ
        self.score_mlp = nn.Sequential(nn.Linear(z_dim * 2, hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(hidden_dim, 1))
        if self.bin_succ:
            self.succ_mlp = nn.Sequential(nn.Linear(z_dim * 2, hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(hidden_dim, hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(hidden_dim, hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(hidden_dim, 2))
            self.succ_criterion = nn.CrossEntropyLoss()
        else:
            self.succ_mlp = nn.Sequential(nn.Linear(z_dim * 2, hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(hidden_dim, hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(hidden_dim, hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(hidden_dim, 1))
            self.succ_criterion = nn.MSELoss()
        self.softmax = nn.Softmax()
        self.score_criterion = nn.MSELoss(reduction='none')
        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, type, eval, detach_encoder=False):
        if type == 'succ':
            if self.args.bin_succ:
                pred = self.succ_mlp(obs)
                if eval:
                    pred = self.softmax(pred)
                    pred = pred[:, 1]  # Probability of success, i.e. being in category 1
            else:
                pred = self.succ_mlp(obs)[:, 0]  # Flatten last dim
                # Clamping makes the gradient zero. Use soft clipping instead
                if 'soft_clipping' in self.args.__dict__ and self.args.soft_clipping:
                    pred = softclipping(pred, 0., 1.)
                else:
                    if eval:
                        pred = torch.clamp(pred, 0., 1.)

        elif type == 'score':
            pred = self.score_mlp(obs)[:, 0]
        return pred

    def log(self, logger, step):
        raise NotImplementedError
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_actor/{k}_hist', v, step)
        for i, m in enumerate(self.trunk):
            if type(m) == nn.Linear:
                logger.log_param(f'train_actor/fc{i}', m, step)


class Agent(object):
    def __init__(self, args, solver, obs_shape, action_dim, num_tools, device='cuda'):
        """obs_shape should be img_size x img_size x img_channel"""
        self.args = args
        self.solver = solver
        obs_shape = (obs_shape[0], obs_shape[1], obs_shape[2] + obs_shape[2] * args.frame_stack)
        print('class Agent: obs_shape: ', obs_shape)
        self.obs_shape = obs_shape
        self.num_tools = num_tools
        self.actors = nn.ModuleList([Actor(args, obs_shape, action_dim).to(device) for _ in range(num_tools)])
        self.feas = nn.ModuleList([FeasibilityPredictor(args, z_dim=args.z_dim).to(device) for _ in range(num_tools)])
        self.vae = VAE(image_channels=len(args.img_mode), z_dim=args.z_dim).to(device)

        self.optim = Adam(list(self.actors.parameters()) + list(self.feas.parameters()) + list(self.vae.parameters()), lr=args.il_lr)
        self.criterion = torch.nn.MSELoss()
        self.terminate_early = True
        self.device = device

    def act(self, obs, goal, tid):
        """ Batch inference! """
        obs = torch.cat([obs, goal], dim=1)  # concat the channel
        action, done = self.actors[tid](obs)
        return action, done

    def fea_pred(self, obs, goal, tid, type, eval=False):
        assert len(obs.shape) == 2  # Make sure the input is the latent code instead of images
        batch_size = 2048
        if len(obs) <= batch_size:
            obs = torch.cat([obs, goal], dim=1)  # concat the channel
            pred = self.feas[tid](obs, type=type, eval=eval)
            return pred
        else:
            N = len(obs)
            all_pred = []
            for i in range(0, N, batch_size):
                obs_cat = torch.cat([obs[i:min(i + batch_size, N)], goal[i:min(i + batch_size, N)]], dim=1)
                pred = self.feas[tid](obs_cat, type=type, eval=eval)
                all_pred.append(pred)

            return torch.cat(all_pred, dim=0)

    def train(self, data_batch, agent_ids=None):
        obses, goals, succ_goals, dones, actions, succ_labels, score_labels, hindsight_flags = \
            data_batch['obses'], data_batch['goal_obses'], data_batch['succ_goals'], data_batch['dones'], data_batch['actions'], \
            data_batch['succ_labels'], data_batch['score_labels'], data_batch['hindsight_flags']

        action_losses, done_losses, succ_losses, score_losses = [], [], [], []
        vae_losses, vae_bce_losses, vae_kl_losses = [], [], []
        tids = range(2)
        if agent_ids is None:
            agent_ids = range(2)
        for tid, agent_id  in zip(tids, agent_ids):
            if 'obs_noise' in self.args.__dict__ and self.args.obs_noise > 0:
                obses[tid] = (torch.rand(obses[tid].shape, device=obses[tid].device) - 0.5) * 2 * self.args.obs_noise + obses[tid]
                goals[tid] = (torch.rand(goals[tid].shape, device=obses[tid].device) - 0.5) * 2 * self.args.obs_noise + goals[tid]
                succ_goals[tid] = (torch.rand(succ_goals[tid].shape, device=obses[tid].device) - 0.5) * 2 * self.args.obs_noise + succ_goals[tid]

            pred_actions, pred_dones = self.act(obses[tid], goals[tid], agent_id)
            action_loss = self.criterion(pred_actions, actions[tid])
            done_loss = self.criterion(pred_dones, dones[tid][:, None])

            z_obs, mu_obs, logvar_obs = self.vae.encode(obses[tid])
            z_goal, mu_goal, logvar_goal = self.vae.encode(goals[tid])
            z_succ_goal, mu_succ_goal, logvar_succ_goal = self.vae.encode(succ_goals[tid])

            z_obs_stop, z_goal_stop, z_succ_goal_stop = z_obs.detach(), z_goal.detach(), z_succ_goal.detach()

            if self.args.back_prop_encoder:
                pred_succ = self.fea_pred(z_obs, z_succ_goal, agent_id, type='succ', eval=False)
            else:
                pred_succ = self.fea_pred(z_obs_stop, z_succ_goal_stop, agent_id, type='succ', eval=False)

            if self.args.back_prop_encoder:
                pred_score = self.fea_pred(z_obs, z_goal, agent_id, type='score', eval=False)
            else:
                pred_score = self.fea_pred(z_obs_stop, z_goal_stop, agent_id, type='score', eval=False)
            if self.args.bin_succ:
                succ_loss = self.feas[agent_id].succ_criterion(pred_succ, succ_labels[tid].flatten().long())
            else:
                succ_loss = self.feas[agent_id].succ_criterion(pred_succ, succ_labels[tid].flatten().float())
            score_loss = self.feas[agent_id].score_criterion(pred_score, score_labels[tid].flatten())  # Not reduced
            score_loss = torch.sum(score_loss * (1. - hindsight_flags[tid])) / torch.sum(
                (1. - hindsight_flags[tid]))  # No score loss for hindsight goals

            # Training VAE using all

            N = len(z_obs) + len(z_goal) + len(z_succ_goal)
            sample_idx = np.random.choice(range(N), size=len(z_obs), replace=False)
            all_z = torch.cat([z_obs, z_goal, z_succ_goal], dim=0)[sample_idx]
            all_mu = torch.cat([mu_obs, mu_goal, mu_succ_goal], dim=0)[sample_idx]
            all_logvar = torch.cat([logvar_obs, logvar_goal, logvar_succ_goal], dim=0)[sample_idx]
            all_original = torch.cat([obses[tid], goals[tid], succ_goals[tid]], dim=0)[sample_idx]
            all_reconstr = self.vae.decode(all_z)
            vae_loss, vae_bce_loss, vae_kl_loss = self.vae.loss_fn(all_reconstr, all_original, all_mu, all_logvar, beta=self.args.encoder_beta)

            vae_losses.append(vae_loss)
            vae_bce_losses.append(vae_bce_loss)
            vae_kl_losses.append(vae_kl_loss)
            action_losses.append(action_loss)
            done_losses.append(done_loss)
            succ_losses.append(succ_loss)
            score_losses.append(score_loss)

        l = sum(action_losses) + sum(done_losses) + sum(vae_losses) + sum(succ_losses) + sum(score_losses)

        self.optim.zero_grad()
        l.backward()
        self.optim.step()
        return {'avg_action_loss': (sum(action_losses) / len(action_losses)).item(),
                'avg_done_loss': (sum(done_losses) / len(done_losses)).item(),
                'avg_vae_loss': (sum(vae_losses) / len(vae_losses)).item(),
                'avg_vae_kl_loss': (sum(vae_kl_losses) / len(vae_kl_losses)).item(),
                'avg_vae_bce_loss': (sum(vae_bce_losses) / len(vae_bce_losses)).item(),
                'avg_score_loss': (sum(score_losses) / len(score_losses)).item(),
                'avg_succ_loss': (sum(succ_losses) / len(succ_losses)).item()}

    def save(self, path):
        torch.save({'actors': self.actors.state_dict(), 'feas': self.feas.state_dict(),
                    'vae': self.vae.state_dict()}, path)

    def load(self, path):
        ckpt = torch.load(path)
        self.actors.load_state_dict(ckpt['actors'])
        self.feas.load_state_dict(ckpt['feas'])
        self.vae.load_state_dict(ckpt['vae'])
        print('Agent loaded from ' + path)

    def load_actor(self, path):
        ckpt = torch.load(path)
        self.actors.load_state_dict(ckpt['actors'])
        print('Actors loaded from ', path)
