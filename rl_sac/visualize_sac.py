import torch.multiprocessing as mp

try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass

import numpy as np
import torch
import argparse
import os
from imitation.env_spec import get_tool_spec
from imitation.eval_helper import get_eval_traj
from plb.utils.visualization_utils import make_grid, save_numpy_as_gif

import tqdm
import json
from chester import logger
from rl_sac.sac import SAC
from rl_sac.run_sac import eval_policy


def eval_sac(args, env):
    spec = get_tool_spec(env, args.env_name)
    args.action_mask = spec['action_masks'][args.tool_combo_id]
    args.contact_loss_mask = np.array(spec['contact_loss_masks'][args.tool_combo_id])
    args.contact_loss_mask_tensor = torch.FloatTensor(args.contact_loss_mask).to('cuda')
    args.discount = float(args.gamma)

    obs_channel = len(args.img_mode) * args.frame_stack
    obs_shape = (args.image_dim, args.image_dim, obs_channel)
    action_dim = env.getattr('action_space.shape[0]', 0)
    max_action = float(env.getattr('env.action_space.high[0]', 0))

    # Agent
    kwargs = {'args': args,
              "obs_shape": obs_shape,
              "action_dim": action_dim,
              "max_action": max_action}

    policy = SAC(**kwargs)
    policy.load(os.path.join(args.agent_path, f'model_{str(args.resume_epoch)}'))

    args.reward_fn = None

    eval_policy(args, policy, env, args.seed, tag=f'{args.env_name}-{args.tool_combo_id}', separate=True, dump_traj_dir=logger.get_dir())
    logger.dump_tabular()
    env.close()


def run_task(arg_vv, log_dir, exp_name):  # Chester launch
    if arg_vv['task'] == 'eval':
        from rl_sac.run_sac import get_args
        from plb.envs.mp_wrapper import make_mp_envs
        agent_path, resume_epoch = arg_vv['sac_agent_path']
        args = get_args()

        vv_path = os.path.join(agent_path, 'variant.json')
        with open(vv_path, 'r') as f:
            vv = json.load(f)
        args.__dict__.update(**vv)
        args.agent_path, args.resume_epoch = agent_path, resume_epoch
        args.num_env = arg_vv['num_env']

        # Configure logger
        logger.configure(dir=log_dir, exp_name=exp_name)
        log_dir = logger.get_dir()
        assert log_dir is not None
        os.makedirs(log_dir, exist_ok=True)
        print("number of devices in current env ", torch.cuda.device_count())

        # Dump parameters
        with open(os.path.join(logger.get_dir(), 'variant.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=2, sort_keys=True)

        # ----------preparation done------------------
        env = make_mp_envs(args.env_name, args.num_env, args.seed)
        args.cached_state_path = env.getattr('cfg.cached_state_path', 0)
        eval_sac(args, env)
    elif arg_vv['task'] == 'demo':
        from rl_sac.run_sac import get_args
        args = get_args()
        vv_path = os.path.join(arg_vv['traj_folder'], 'variant.json')

        with open(vv_path, 'r') as f:
            vv = json.load(f)
        args.__dict__.update(**vv)
        args.num_env = arg_vv['num_env']
        args.img_size = arg_vv['img_size']

        from plb.envs import make
        from imitation.env_spec import set_render_mode
        from iclr.replay import replay
        from glob import glob

        # Need to make the environment before moving tensor to torch
        env = make(args.env_name)
        set_render_mode(env, args.env_name)

        traj_folder = arg_vv['traj_folder']
        for traj_path in glob(os.path.join(traj_folder, '*.pkl')):
            replay(args, env, traj_path, save_folder=traj_folder)
        env.close()
