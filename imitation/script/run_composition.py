# %%
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
from imitation.agent import Agent
from imitation.sampler import sample_traj
from plb.envs import make
from imitation.train import get_args
from plb.engine.taichi_env import TaichiEnv
from plb.optimizer.solver import Solver
from plb.algorithms.logger import Logger

device = 'cuda'

log_dir = './data/connect'
args = get_args("")

obs_channel = len(args.img_mode)
img_obs_shape = (args.image_dim, args.image_dim, obs_channel)

env = make(args.env_name, nn=(args.algo == 'nn'), sdf_loss=args.sdf_loss,
           density_loss=args.density_loss, contact_loss=args.contact_loss,
           soft_contact_loss=args.soft_contact_loss, chamfer_loss=args.chamfer_loss)
env.seed(args.seed)
taichi_env: TaichiEnv = env.unwrapped.taichi_env
T = env._max_episode_steps
action_dim = taichi_env.primitives.action_dim

plb_logger = Logger(log_dir)
solver = Solver(taichi_env, plb_logger, None,
                n_iters=(args.gd_num_steps + T - 1) // T, softness=args.softness, horizon=T,
                **{"optim.lr": args.lr, "optim.type": args.optim, "init_range": 0.0001})
agent = Agent(args, solver, img_obs_shape, action_dim, num_tools=2, device=device)

from imitation.utils import load_target_info

target_info = load_target_info(args, device)
np_target_imgs, target_imgs, np_target_mass_grids = target_info['np_target_imgs'], target_info['target_imgs'], target_info['np_target_mass_grids']

import torch
import matplotlib.pyplot as plt
from imitation.utils import img_to_np, write_number, img_to_tensor
from plb.utils.visualization_utils import make_grid, save_rgb
import os.path as osp
import sys
import json

sys.path.insert(0, '/home/xingyu/Projects/PlasticineLab/')

# Trained on hindsight goals
# agent_path = 'data/local/0629_PushSpread_train_fea/0629_PushSpread_train_fea_2021_07_01_11_34_12_0001/agent_350.ckpt'
# Trained on real goals
# agent_path = 'data/autobot/0805_PushSpread_fea/0805_PushSpread_fea/0805_PushSpread_fea_2021_08_05_16_50_05_0002/agent_50.ckpt'
# agent_path = 'data/autobot/0805_PushSpread_fea/0805_PushSpread_fea/0805_PushSpread_fea_2021_08_05_16_50_05_0002/agent_50.ckpt'
agent_path = 'data/autobot/0807_PushSpread_fea/0807_PushSpread_fea/0807_PushSpread_fea_2021_08_07_17_20_51_0001/agent_30.ckpt'
vv_path = osp.join(osp.dirname(agent_path), 'variant.json')
with open(vv_path, 'rb') as f:
    agent_vv = json.load(f)
args.__dict__.update(**agent_vv)
print('bin succ:', args.bin_succ)
agent = Agent(args, solver, img_obs_shape, action_dim, num_tools=2, device=device)
agent.feas.eval()
agent.load(agent_path)
taichi_env.loss.set_target_update(False)

from imitation.buffer import ReplayBuffer

buffer = ReplayBuffer()

import json
from imitation.encoder.vae import VAE

vae_dir = 'data/autobot/0729_vae/0729_vae/0729_vae_2021_07_29_20_03_08_0006/'
vae_path = osp.join(vae_dir, 'encoder_95.pth')
vae_vv_path = osp.join(vae_dir, 'variant.json')
with open(vae_vv_path, 'r') as f:
    vae_vv = json.load(f)
vae = VAE(4)
vae.load_state_dict(torch.load(vae_path))
vae = vae.cuda()
# %%
import os

execute = True
from imitation.compose_skills import plan, visualize_all_traj
from imitation.compose_skills import execute
from imitation.compose_skills import visualize_adam_info, visualize_mgoal

opt_mode = 'adam'  # ['search', 'sample', 'adam']
save_path = f'./data/connect/{opt_mode}'
if not os.path.exists(save_path):
    os.mkdir(save_path)

test_init_v = np.array([95, 95, 95, 91, 91, 91, 80])
test_target_v = np.array([25, 195, 128, 21, 191, 128, 20])

opt_args = {}
if opt_mode == 'search':
    opt_args['mid_goals'] = np_target_imgs[test_target_v]
elif opt_mode == 'sample':
    opt_args['num_sample'] = 10000
    opt_args['vae'] = vae
elif opt_mode == 'adam':
    opt_args['adam_sample'] = 1000
    opt_args['adam_iter'] = 2000
    opt_args['adam_lr'] = 5e-2
    opt_args['min_zlogl'] = float(-50)
    opt_args['vae'] = vae
    opt_args['save_goal_his'] = False
for i, (init_v, target_v) in enumerate(zip(test_init_v, test_target_v)):
    if i not in [1, 2, 5]:
        continue
    plan_info = {'env': env, 'init_v': init_v, 'target_v': target_v}
    plan_info.update(**opt_args)
    plan_info.update(**target_info)
    best_traj, all_traj, traj_info = plan(agent, plan_info, opt_mode=opt_mode)
    img, sorted_idxes = visualize_all_traj(all_traj)
    save_grid_img = make_grid(img[:10], ncol=5, padding=5, pad_value=0.5)
    save_rgb(osp.join(save_path, f'plan_{i}.png'), save_grid_img[:, :, :3])

    for j in range(5):
        execute_name = osp.join(save_path, f'execute_{i}_run_{j}.gif')
        execute(env, agent, all_traj[j], target_info, execute_name)

    if opt_mode == 'adam':
        visualize_adam_info(traj_info, savename=osp.join(save_path, f'adam_{i}.png'), topk=8)
        if 'goal_his' in traj_info:
            goal_his = np.concatenate(traj_info['goal_his'], axis=1)  # traj_info goal his: [4 (list), 200, 10, img_dims]
            goal_his = goal_his[:, sorted_idxes[:10]]
            visualize_mgoal(goal_his, savename=osp.join(save_path, f'adam_{i}.gif'))
