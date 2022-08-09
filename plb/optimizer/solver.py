import numpy as np
from imitation.utils import rotate_y_axis_angle
import plb.utils.utils as tu
import torch
import tqdm
import os
from chester import logger
import wandb
# tu.set_default_tensor_type(torch.DoubleTensor)
from ..engine.function import GradModel

FUNCS = {}


class Solver:
    def __init__(self, args, env, ouput_grid=(), **kwargs):
        self.args = args
        self.num_resets = args.num_resets
        self.horizon = args.buffer_horizon
        self.reset_ts = []
        self.reset_angs = []
        if args.opt_mode == 'multireset':
            self.reset_ts = [self.horizon // (self.num_resets + 1) * i for i in range(1, self.num_resets+1)]
            self.reset_angs = [np.pi / (self.num_resets + 1) * i for i in range(1, self.num_resets+1)]
        print("reset timesteps:", self.reset_ts)
        print("reset angles:", self.reset_angs)
        self.env = env
        if env not in FUNCS:
            FUNCS[env] = GradModel(env, output_grid=ouput_grid, **kwargs)
        self.func = FUNCS[env]
        self.buffer = []

    def solve(self, initial_actions, loss_fn, action_mask=None, lr=0.01, max_iter=200, verbose=True, scheduler=None, debug=False):
        # loss is a function of the observer ..
        if action_mask is not None:
            initial_actions = initial_actions * action_mask[None]
            action_mask = torch.FloatTensor(action_mask[None]).cuda()
        best_action, best_loss = initial_actions, np.inf
        self._buffer = []
        self.buffer.append(self._buffer)
        self.action = action = torch.nn.Parameter(tu.np2th(np.array(initial_actions)), requires_grad=True)
        self.optim = optim = torch.optim.Adam([action], lr=lr)
        self.reset_optim = torch.optim.Adam([action], lr=self.args.reset_lr)
        # self.optim = optim = torch.optim.LBFGS([action], history_size=1)
        self.initial_state = initial_state = self.env.get_state()
        self.scheduler = scheduler if scheduler is None else scheduler(self.optim)
        iter_id = 0
        ran = tqdm.trange if verbose else range
        it = ran(iter_id, iter_id + max_iter)
        loss, last = np.inf, initial_actions
        reset_loss = False
        init_primitive_states = self.initial_state['state'][-len(self.env.primitives):]
        print("init_primitive_states:", init_primitive_states)
        if self.args.debug_gradient:
            self.grads = []
            self.actions = []
        for iter_id in it:
            if self.args.opt_mode == 'twopass' and iter_id > 0 and iter_id % self.args.reset_loss_freq == 0:
                self.optimize_reset(initial_state, action, loss_fn, init_primitive_states[:7], verbose, iter_id)
            optim.zero_grad()
            observations = self.func.reset(initial_state['state'])
            cached_obs = []
            for idx, i in enumerate(action):
                if self.args.opt_mode == 'manual' and self.args.eps_length ==170:  # manual reset module max_step=170
                    if idx < 50 or idx >=120:
                        observations = self.func.forward(idx, i, None, *observations)
                        cached_obs.append(observations)
                    elif idx <100:
                        reset_action = (-1*action[50-(idx-49)]).detach()
                        if idx >= 80:
                            reset_action[2] = 0. 
                        observations = self.func.forward(idx, reset_action, None, *observations)
                    else:
                        reset_action = torch.tensor([0, 1, 0, 0, 0, 0], dtype=torch.float).cuda()
                        observations = self.func.forward(idx, reset_action, None, *observations)
                elif self.args.opt_mode == 'twopass': # two pass gradients max_step=120
                    reset_loss= False
                    if idx < 50 or idx >= 100:
                        observations = self.func.forward(idx, i, None, *observations)
                        cached_obs.append(observations)
                    else:
                        observations = self.func.forward(idx, i.detach(), None, *observations)
                elif self.args.opt_mode == 'multireset' and idx in self.reset_ts:
                    reset_idx = self.reset_ts.index(idx)
                    new_reset_states = rotate_y_axis_angle(init_primitive_states, 0, self.reset_angs[reset_idx])
                    new_reset_states[0][1] -= 0.1
                    observations = self.func.forward(idx, i, new_reset_states, *observations)    
                    cached_obs.append(observations)
                else:
                    observations = self.func.forward(idx, i, None, *observations)
                    cached_obs.append(observations)
                    
            loss, all_losses = loss_fn(list(range(len(cached_obs))), cached_obs, self.args.vel_loss_weight, \
                reset_loss=reset_loss, reset_state=init_primitive_states[:7])
            loss.backward()
            if self.args.debug_gradient:
                self.grads.append(action.grad.cpu().detach().numpy())
                self.actions.append(action.cpu().detach().numpy())
            optim.step()

            if self.scheduler is not None:
                self.scheduler.step()
            action.data.copy_(torch.clamp(action.data, -1, 1))
            if action_mask is not None:
                action.data.copy_(action.data * action_mask)

            with torch.no_grad():
                loss = loss.item()
                losses = [l.detach().cpu().numpy() for l in all_losses]
                last = action.data.detach().cpu().numpy()
                if self.args.opt_mode=='manual' and self.args.eps_length == 170:
                    new_last = -1*last[49::-1]
                    new_last[30:, 2:3] *= 0.
                    rotation = np.tile(np.array([0., 1., 0., 0., 0., 0.]), (20, 1))
                    last = np.vstack([last[:50], new_last, rotation, last[120:]])
                if not reset_loss and loss < best_loss:
                    best_loss = loss
                    best_action = last

            self._buffer.append({'action': last, 'loss': loss, 'all_losses': np.array(losses)})
            if verbose:
                it.set_description(f"Itr {iter_id}: EMD loss {loss}", refresh=True)

            if iter_id % 10 == 0:
                self.dump_buffer(os.path.join(logger.get_dir(), 'buffer.pkl'))

        self.dump_buffer(os.path.join(logger.get_dir(), 'buffer.pkl'))
        self.env.set_state(**initial_state)
        ret = {
            'best_loss': best_loss,
            'best_action': best_action,
            'last_loss': loss,
            'last_action': last
        }
        if self.args.debug_gradient:
            ret['grads'] = self.grads
            ret['actions'] = self.actions
        return ret

    def optimize_reset(self, initial_state, action, loss_fn, reset_state, verbose, iter_id):
        reset_loss = True
        it_reset = tqdm.trange(self.args.reset_loss_steps)
        for reset_itr in it_reset:
            self.reset_optim.zero_grad()
            observations = self.func.reset(initial_state['state'])
            cached_obs = []
            for idx, i in enumerate(action):
                if idx < 50 or idx >= 100:
                    observations = self.func.forward(idx, i.detach(), None, *observations)
                else:
                    observations = self.func.forward(idx, i, None, *observations)
                    cached_obs.append(observations)
            loss, _ = loss_fn(list(range(len(cached_obs))), cached_obs, self.args.vel_loss_weight, \
            reset_loss=reset_loss, reset_state=reset_state)
            loss.backward()
            self.reset_optim.step()
            with torch.no_grad():
                loss = loss.item()
                last = action.data.detach().cpu().numpy()
                self._buffer.append({'action': last, 'reset_loss': loss})
                if verbose:
                    it_reset.set_description(f"Itr {iter_id}, Reset {reset_itr}: Reset loss {loss}", refresh=True)


    def plot_grad(self, path):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1)
        grads = np.array(self.grads)
        ax.plot(grads[:, 0], label="dim0")
        ax.plot(grads[:, 1], label="dim1")
        ax.plot(grads[:, 2], label="dim2")
        ax.plot(grads[:, 3], label="dim3")
        # plt.plot(grads[:, 4], label="dim4")
        # plt.plot(grads[:, 5], label="dim5")
        ax.legend()
        plt.savefig(path)
        plt.close()

    def eval(self, action, initial_state, render_fn):
        self.env.simulator.cur = 0
        self.env.set_state(**initial_state)
        outs = []
        import tqdm
        for i in tqdm.tqdm(action, total=len(action)):
            self.env.step(i)
            outs.append(render_fn())

        self.env.set_state(**initial_state)
        return outs

    def plot_buffer(self):
        import matplotlib.pyplot as plt
        plt.Figure()
        losses = []
        loss_idx = []
        reset_losses = []
        reset_loss_idx = []
        for i in range(len(self._buffer)):
            if 'reset_loss' in self._buffer[i].keys():
                reset_loss_idx.append(i)
                reset_losses.append(self._buffer[i]['reset_loss'])
            else:
                loss_idx.append(i)
                losses.append(self._buffer[i]['loss'])
        plt.plot(loss_idx, losses, 'EMD Loss')
        plt.plot(reset_loss_idx, reset_losses, 'Reset Loss')
        plt.legend()
        plt.show()

    def save_plot_buffer(self, path, postfix=''):
        self.save_plot_emd(os.path.join(path, f"emd_loss{postfix}.png"))
        self.save_plot_reset(os.path.join(path, f"reset_loss{postfix}.png"))

    def save_plot_emd(self, path):
        import matplotlib.pyplot as plt
        plt.Figure()
        for buf in self.buffer:
            losses = []
            loss_idx = []
            for i in range(len(buf)):
                losses.append(buf[i]['loss'])
                loss_idx.append(i)
            plt.plot(loss_idx, losses)
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.savefig(path)
        plt.close()

    def save_plot_reset(self, path):
        import matplotlib.pyplot as plt
        plt.Figure()
        for buf in self.buffer:
            losses = []
            loss_idx = []
            for i in range(len(buf)):
                if 'reset_loss' in buf[i].keys():
                    losses.append(buf[i]['reset_loss'])
                    loss_idx.append(i)
            plt.plot(loss_idx, losses)
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.savefig(path)
        plt.close()

    def dump_buffer(self, path='/tmp/buffer.pkl'):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, path='/tmp/buffer.pkl'):
        import pickle
        with open(path, 'rb') as f:
            self.buffer = pickle.load(f)
