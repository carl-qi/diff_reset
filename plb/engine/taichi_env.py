import numpy as np
import cv2
import taichi as ti
from taichi.lang.impl import reset
import torch
import os
import pykeops
from plb.engine.primitive import primitives
from plb.engine.primitive.primitives import Gripper
from scipy.spatial.transform import Rotation as R

myhost = os.uname()[1]
dir = os.path.join(os.path.expanduser("~"), '.pykeops_cache/'+torch.cuda.get_device_name(0).replace(' ', '_'))
os.makedirs(dir, exist_ok=True)
print('Setting pykeops dir to ', dir)
pykeops.set_bin_folder(dir)

# TODO: run on GPU, fast_math will cause error on float64's sqrt; removing it cuases compile error..
ti.init(arch=ti.gpu, debug=False, fast_math=True)


def create_emd_loss():
    from geomloss import SamplesLoss
    loss = SamplesLoss(loss='sinkhorn', p=1, blur=0.001)
    return loss


def create_emd_loss_small_mem():
    from geomloss import SamplesLoss
    loss = SamplesLoss(loss='sinkhorn', p=1, blur=0.001)
    return loss


@ti.data_oriented
class TaichiEnv:
    def __init__(self, cfg, nn=False, loss=True, return_dist=False):
        """
        A taichi env builds scene according the configuration and the set of manipulators
        """
        # primitives are environment specific parameters ..
        # move it inside can improve speed; don't know why..
        from .mpm_simulator import MPMSimulator
        from .primitive import Primitives
        from .renderer import Renderer
        from plb.env_modeling.tina_renderer import TinaRenderer
        from .shapes import Shapes
        from .losses import Loss
        from .nn.mlp import MLP

        self.has_loss = loss
        self.cfg = cfg.ENV
        self.primitives = Primitives(cfg.PRIMITIVES)
        self.shapes = Shapes(cfg.SHAPES)
        self.init_particles, self.particle_colors = self.shapes.get()

        cfg.SIMULATOR.defrost()
        self.n_particles = cfg.SIMULATOR.n_particles = len(self.init_particles)

        self.simulator = MPMSimulator(cfg.SIMULATOR, self.primitives)
        self.dim = self.simulator.dim
        if 'name' not in cfg.RENDERER.__dict__ or cfg.RENDERER.name == 'tina':
            self.renderer = TinaRenderer(cfg.RENDERER, self.primitives)
            self.renderer_name = 'tina'
        if cfg.RENDERER.name == 'plb':
            raise NotImplementedError
            self.renderer = Renderer(cfg.RENDERER, self.primitives)
            self.renderer_name = 'plb'

        if nn:
            self.nn = MLP(self.simulator, self.primitives, (256, 256))

        self._is_copy = True

        self.target_x, self.tensor_target_x = None, None
        self.device = 'cuda'
        self.contact_loss_mask = None
        self.return_dist = return_dist
        if self.return_dist:
            self.dists = []
            self.dists_start_idx = []
            for i in self.primitives:
                self.dists_start_idx.append(len(self.dists))
                if isinstance(i, Gripper):
                    self.dists += [ti.field(dtype=self.simulator.dtype, shape=(self.simulator.n_particles,), needs_grad=True)]
                self.dists += [ti.field(dtype=self.simulator.dtype, shape=(self.simulator.n_particles,), needs_grad=True)]
            self.dists_start_idx.append(len(self.dists))


    def set_copy(self, is_copy: bool):
        self._is_copy = is_copy

    def initialize(self, cfg=None, target_path=None):
        if cfg is not None:
            from .shapes import Shapes
            from .primitive import Primitives
            self.cfg = cfg
            self.shapes = Shapes(self.cfg.SHAPES)
            self.init_particles, self.particle_colors = self.shapes.get()
            self.primitives.update_cfgs(cfg.PRIMITIVES)
            if self.has_loss and target_path is not None:
                self.load_target_x(target_path)

        # initialize all taichi variable according to configurations..
        self.primitives.initialize()
        self.simulator.initialize()
        self.renderer.initialize(self.cfg.RENDERER)

        # call set_state instead of reset..
        self.simulator.reset(self.init_particles)
        self.init_particles, self.particle_colors = self.shapes.get()
        # self.sampled_idx = sampled_idx = np.random.choice(xs[0].shape[0], 500, replace=False)

    def load_target_x(self, path):
        self.target_x = np.load(path)
        self.tensor_target_x = torch.FloatTensor(self.target_x).to(self.device)

    def render(self, mode='human', **kwargs):
        assert self._is_copy, "The environment must be in the copy mode for render ..."
        if self.renderer_name == 'plb':
            if self.n_particles > 0:
                x = self.simulator.get_x(0)
                self.renderer.set_particles(x, self.particle_colors)
            img = self.renderer.render_frame(**kwargs)
            rgb = np.uint8(img[:, :, :3].clip(0, 1) * 255)

            if mode == 'human':
                cv2.imshow('x', rgb[..., ::-1])
                cv2.waitKey(1)
            elif mode == 'plt':
                import matplotlib.pyplot as plt
                plt.imshow(rgb)
                plt.show()
            else:
                return img
        elif self.renderer_name == 'tina':
            if self.n_particles > 0:
                x = self.simulator.get_x(0)
                self.renderer.set_particles(x, self.particle_colors)
            img = self.renderer.render(mode=mode, **kwargs)
            return img
        else:
            raise NotImplementedError

    def step(self, action=None, state=None):
        if action is not None:
            action = np.array(action)
        self.simulator.step(is_copy=self._is_copy, action=action, state=state)

    def get_state(self):
        assert self.simulator.cur == 0
        return {
            'state': self.simulator.get_state(0),
            'softness': self.primitives.get_softness(),
            'is_copy': self._is_copy
        }

    def set_state(self, state, softness=None, is_copy=None, **kwargs):
        if softness is None:
            softness = state['softness']
            is_copy = state['is_copy']
            state = state['state']

        self.simulator.cur = 0
        self.simulator.set_state(0, state)
        self.primitives.set_softness(softness)
        self._is_copy = is_copy

    def set_primitive_state(self, state, softness, is_copy, **kwargs):
        self.simulator.set_primitive_state(0, state)

    def get_use_gripper_primitive(self):
        if not hasattr(self, 'use_gripper_primitive'):
            from plb.engine.primitive.primitives import Gripper
            self.use_gripper_primitive = False
            for i in self.primitives.primitives:
                if isinstance(i, Gripper):
                    self.use_gripper_primitive = True
        return self.use_gripper_primitive

    def get_contact_loss(self, shape):
        if self.get_use_gripper_primitive():
            dists = shape[:, -len(self.primitives) - 1:]  # manipulator distance
            gripper_dist = dists[:, 0] + dists[:, 1]
            dists = torch.cat([gripper_dist[:, None], dists[:, 2:]], dim=1)
        else:
            dists = shape[:, -len(self.primitives):]  # manipulator distance
        min_ = dists.min(axis=0)[0].clamp(0, 1e9)
        assert self.contact_loss_mask.shape == min_.shape
        min_ = self.contact_loss_mask * min_
        dist = (min_ ** 2).sum() * 1e-3
        return dist

    def compute_loss(self, idxes, observations, vel_loss_weight, reset_loss=False, reset_state=None):  # shape: n x (6 + k), first 3 position, 3 velocity, dist to manipulator, k: number of manipulator
        """ Loss for traj_opt"""
        loss = 0
        all_losses = []


        if reset_loss:
            assert reset_state is not None
            loss += self.compute_loss_reset(observations, reset_state)
            return loss, 0

        if not hasattr(self, 'loss_fn'):
            self.loss_fn = create_emd_loss_small_mem()

        xs = []
        for idx, (shape, tool, *args) in zip(idxes, observations):
            dist = self.get_contact_loss(shape)
            loss += dist
            all_losses.append(dist)
            xs.append(shape[:, :3])
        contact_loss = loss.item()
        sampled_idx = np.random.choice(xs[0].shape[0], 500, replace=False)
        target_x = self.tensor_target_x[sampled_idx].repeat([len(xs), 1, 1])
        xs = torch.stack(xs).contiguous()[:, sampled_idx]
        for i in range(len(xs)):
            curr_loss = self.loss_fn(xs[i], target_x[i])
            loss += curr_loss
            all_losses[-1] += curr_loss
        final_v = observations[-1][0]
        loss += vel_loss_weight * torch.sum(torch.mean(final_v[:, 3:6], dim=0) ** 2)
        return loss, all_losses


    def compute_loss_reset(self, observations, reset_state):  # shape: n x (6 + k), first 3 position, 3 velocity, dist to manipulator, k: number of manipulator
        """ Loss for traj_opt"""
        state_copy = reset_state
        p = R.from_quat([state_copy[4], state_copy[5], state_copy[6], state_copy[3]])
        q = R.from_rotvec(np.pi/2 * np.array([0, 1, 0])) # hard coded rotate 90% around y-axis for now
        quat = R.as_quat(q*p)
        reset_state = torch.tensor([state_copy[0], state_copy[1]-0.1, state_copy[2], quat[-1], quat[0], quat[1], quat[2]]).cuda()
        loss = torch.mean((observations[-1][1][0][:7] - reset_state) ** 2)
        return loss
    # def her_reward_fn(self, tool_state, particle_state, goal_particle_state):  # Redo for RL
    #     if not hasattr(self, 'loss_fn'):
    #         self.loss_fn = create_emd_loss()
    #     if not isinstance(tool_state, torch.Tensor):
    #         tool_state = torch.FloatTensor(tool_state).to('cuda', non_blocking=True)
    #         particle_state = torch.FloatTensor(particle_state).to('cuda', non_blocking=True)
    #         goal_particle_state = torch.FloatTensor(goal_particle_state).to('cuda', non_blocking=True)
    #
    #     emd = self.loss_fn(particle_state, goal_particle_state)
    #     dists = torch.min(torch.cdist(tool_state[:, :3][None], particle_state[None])[0], dim=1)[0]
    #     contact_loss = (self.contact_loss_mask * dists).sum() * 1e-3
    #     reward = -emd - contact_loss
    #     return reward.item()

    def set_init_emd(self):
        self.init_emd = self.get_curr_emd()
        if isinstance(self.init_emd, torch.Tensor):
            self.init_emd = self.init_emd.item()

    def get_curr_emd(self):
        if self.tensor_target_x is None:
            print("Generating target, skipping setting emd")
            return 0.
        if not hasattr(self, 'loss_fn'):
            self.loss_fn = create_emd_loss()
        curr_x = self.simulator.get_torch_x(0, self.device).contiguous()
        emd = self.loss_fn(curr_x, self.tensor_target_x)
        return emd

    def get_reward_and_info(self):
        if self.tensor_target_x is None:
            # Generating target
            return 0., {}
        else:
            if not hasattr(self, 'loss_fn'):
                self.loss_fn = create_emd_loss()
            curr_x = self.simulator.get_torch_x(0, self.device).contiguous()
            sampled_idx = np.random.choice(curr_x.shape[0], 500, replace=False)
            curr_x = curr_x[sampled_idx]
            target_x = self.tensor_target_x[sampled_idx]
            emd = self.loss_fn(curr_x, target_x)
            primitive_state = torch.cat([i.get_state_tensor(0)[None, :3] for i in self.primitives], dim=0)
            dists = torch.min(torch.cdist(primitive_state[None], curr_x[None])[0], dim=1)[0] # find the nearest point on the dough to the center of the primitives
            contact_loss = (self.contact_loss_mask * dists).sum() * 1
            reward = -emd - contact_loss
            emd = emd.item()
            if not hasattr(self, 'init_emd'):
                normalized_performance = 0.
            elif self.init_emd == 0.:
                normalized_performance = 1.
            else:
                normalized_performance = (self.init_emd - emd) / self.init_emd
            info = {'info_emd': emd,
                    'info_normalized_performance': normalized_performance,
                    'info_contact_loss': contact_loss.item()}
            return reward.item(), info

    def get_geom_loss_fn(self):
        if not hasattr(self, 'loss_fn'):
            self.loss_fn = create_emd_loss()
        return self.loss_fn

    def set_contact_loss_mask(self, mask):
        self.contact_loss_mask = mask


    @ti.kernel
    def _get_obs(self, s: ti.int32, x: ti.ext_arr(), c: ti.ext_arr(), p: ti.ext_arr()):
        for i in range(self.simulator.n_particles):
            for j in ti.static(range(self.dim)):
                x[i, j] = self.simulator.x[s, i][j]
                # x[i, j + self.dim] = self.simulator.v[s, i][j]
        for idx, i in ti.static(enumerate(self.primitives)):
            for j in ti.static(range(i.pos_dim)):
                c[idx, j] = i.position[s][j]
            for j in ti.static(range(i.rotation_dim)):
                c[idx, j + i.pos_dim] = i.rotation[s][j]
            if ti.static(i.state_dim == 8):
                c[idx, 7] = i.gap[s]
            for j in ti.static(range(i.num_rand_points)):
                for d in ti.static(range(self.dim)):
                    p[idx, j, d] = i.rand_points[s, j][d]

    def get_obs(self, s, device):
        f = s * self.simulator.substeps
        x = torch.zeros(size=(self.simulator.n_particles, self.dim * 2), device=device)
        x = x[:, :3] # Only the position
        c = torch.zeros(size=(len(self.primitives), 8), device=device)  # hack for gripper
        action_primitives = [prim for prim in self.primitives if prim.action_dim > 0] #only generate particles for actionable tools
        p = torch.zeros(size=(len(action_primitives), self.primitives[0].num_rand_points, 3), device=device) # particles for tools
        self._get_obs(f, x, c, p)
        if self.return_dist: # Not really used since this function should only be called when return_dist is True
            self.compute_min_dist(f)
            dists = torch.cat([i.to_torch(device)[:, None] for i in self.dists], 1)
            merged_dists = []
            for i in range(len(self.dists_start_idx)-1):
                l, r = self.dists_start_idx[i], self.dists_start_idx[i+1]
                merged_dists.append(torch.sum(dists[:, l:r], dim=1, keepdim=True))
            x = torch.cat((x, *merged_dists), 1)
        outputs = x.clone(), c.clone(), p.clone()
        return outputs

    @ti.kernel
    def compute_min_dist(self, f: ti.int32):
        for j in ti.static(range(self.simulator.n_primitive)):
            for i in range(self.simulator.n_particles):
                v = ti.static(self.dists_start_idx[j])
                if ti.static(not isinstance(self.simulator.primitives[j], Gripper)):
                    self.dists[v][i] = self.simulator.primitives[j].sdf(f, self.simulator.x[f, i])
                else:
                    self.dists[v][i] = self.simulator.primitives[j].sdf_2(f, self.simulator.x[f, i], -1)
                    self.dists[v + 1][i] = self.simulator.primitives[j].sdf_2(f, self.simulator.x[f, i], 1)
