from imitation.sampler import sample_traj
from imitation.imitation_buffer import ImitationReplayBuffer
from imitation.args import get_args
from imitation.utils import calculate_performance, visualize_dataset
from imitation.utils import visualize_trajs
from imitation.env_spec import set_render_mode
from plb.envs import make

import random
import numpy as np
import torch
import json
import os
from chester import logger


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def eval_training_traj(traj_ids, args, buffer, env, agent, target_imgs, np_target_imgs, np_target_mass_grids, save_name):
    horizon = 100
    trajs = []
    demo_obses = []
    demo_target_ious = []
    for traj_id in traj_ids:
        init_v = int(buffer.buffer['init_v'][traj_id * horizon])
        target_v = int(buffer.buffer['target_v'][traj_id * horizon])
        reset_key = {'init_v': init_v, 'target_v': target_v}
        tid = buffer.get_tid(buffer.buffer['action_mask'][traj_id * horizon])
        traj = sample_traj(env, agent, reset_key, tid)
        traj['target_img'] = np_target_imgs[reset_key['target_v']]
        demo_obs = buffer.buffer['obses'][traj_id * horizon: traj_id * horizon + horizon]
        demo_obses.append(demo_obs)
        demo_target_ious.append(buffer.buffer['target_ious'][traj_id * horizon + horizon - 1])
        print(f'tid: {tid}, traj_id: {traj_id}, reward: {np.sum(traj["rewards"])}')
        trajs.append(traj)
    demo_obses = np.array(demo_obses)

    agent_ious = np.array([traj['target_ious'][-1, 0] for traj in trajs])
    demo_target_ious = np.array(demo_target_ious)
    logger.log('Agent ious: {}, Demo ious: {}'.format(np.mean(agent_ious), np.mean(demo_target_ious)))

    visualize_trajs(trajs, 4, key='info_normalized_performance', save_name=os.path.join(logger.get_dir(), save_name),
                    vis_target=True, demo_obses=demo_obses[:, :, :, :, :3])
    info = {'agent_iou': np.mean(agent_ious), 'demo_iou': np.mean(demo_target_ious)}
    return trajs, info


def run_task(arg_vv, log_dir, exp_name):  # Chester launch
    from plb.engine.taichi_env import TaichiEnv
    from plb.optimizer.solver import Solver
    args = get_args(cmd=False)
    args.__dict__.update(**arg_vv)

    set_random_seed(args.seed)

    device = 'cuda'

    # Configure logger
    logger.configure(dir=log_dir, exp_name=exp_name)
    log_dir = logger.get_dir()
    assert log_dir is not None
    os.makedirs(log_dir, exist_ok=True)
    # Dump parameters
    with open(os.path.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2, sort_keys=True)

    if args.use_wandb:
        import wandb
        wandb.init(project='Dough',
                   group=args.wandb_group,
                   name=exp_name,
                   resume='allow', id=None,
                   settings=wandb.Settings(start_method='fork'))
        wandb.config.update(args, allow_val_change=True)
    # ----------preparation done------------------
    buffer = ImitationReplayBuffer(args)
    buffer2 = None
    # buffer2 = ImitationReplayBuffer(args)
    # buffer2.load('/home/hanwenq/Projects/PlasticineLab/data/local/1206_Roll_benchmarking/1206_Roll_benchmarking_2021_12_06_15_41_39_0005/dataset.gz')
    # buffer2.load('/home/jianrenw/carl/research/dev/PlasticineLab/data/seuss/1206_Roll_benchmarking/1206_Roll_benchmarking/1206_Roll_benchmarking_2021_12_06_15_41_39_0005/dataset.gz')
    obs_channel = len(args.img_mode) * args.frame_stack
    img_obs_shape = (args.image_dim, args.image_dim, obs_channel)
    env = make(args.env_name, nn=(args.algo == 'nn'))
    env.seed(args.seed)
    taichi_env: TaichiEnv = env.unwrapped.taichi_env
    set_render_mode(env, args.env_name, 'mesh')

    if args.data_name == 'demo':
        traj_ids = np.array_split(np.arange(args.num_trajs), args.gen_num_batch)[args.gen_batch_id]
        print('traj_ids:', traj_ids)

        def get_state_goal_id(traj_id):
            np.random.seed(traj_id+args.seed)
            
            if env.num_targets == 125:
                goal_id = np.random.randint(0, env.num_targets)
                size = goal_id % 5
                state_id = np.random.randint(0, 25)
                state_id = state_id * 5 + size
            else:
                goal_id = np.random.randint(0, env.num_targets)
                state_id = np.random.randint(0, env.num_inits)
            return {'init_v': state_id, 'target_v': goal_id}  # state and target version
    else:
        from imitation.hardcoded_eval_trajs import get_eval_traj
        init_vs, target_vs = get_eval_traj(env.cfg.cached_state_path)

        traj_ids = np.array_split(np.arange(10), args.gen_num_batch)[args.gen_batch_id]
        def get_state_goal_id(traj_id):
            return {'init_v': init_vs[traj_id], 'target_v': target_vs[traj_id]}  # state and target version

    solver = Solver(args, taichi_env, (0,), return_dist=True)
    args.dataset_path = os.path.join(logger.get_dir(), 'dataset.gz')

    from imitation.env_spec import get_tool_spec
    tool_spec = get_tool_spec(env, args.env_name)
    for tid in range(len(tool_spec['contact_loss_masks'])):
        action_mask = tool_spec['action_masks'][tid]
        contact_loss_mask = tool_spec['contact_loss_masks'][tid]
        for i, traj_id in enumerate(traj_ids):
            reset_key = get_state_goal_id(traj_id)
            reset_key['contact_loss_mask'] = contact_loss_mask
            traj = sample_traj(env, solver, reset_key, tid, action_mask=action_mask, reset_primitive=args.reset_primitive, num_moves=args.num_moves, init=args.action_init, buffer=buffer2)
            print(
                f"traj {traj_id}, agent time: {traj['info_agent_time']}, env time: {traj['info_env_time']}, total time: {traj['info_total_time']}")
            buffer.add(traj)
            if i % 1 == 0:
                buffer.save(os.path.join(args.dataset_path))
                visualize_dataset(args.dataset_path, env.cfg.cached_state_path, os.path.join(logger.get_dir(), 'visualization.gif'),
                                  visualize_reset=False,
                                  overlay_target=True,
                                  max_step=args.buffer_horizon,
                                  num_moves=args.num_moves)
                perf = calculate_performance(args.dataset_path, max_step=args.buffer_horizon, num_moves=args.num_moves)
                print("Mean performance so far:", perf)
                if args.use_wandb:
                    wandb.log({'avg performance': perf})
    # from imitation.generate_reset_motion import generate_reset_motion
    # generate_reset_motion(buffer, env)
    buffer.save(os.path.join(args.dataset_path))
    visualize_dataset(args.dataset_path, env.cfg.cached_state_path, os.path.join(logger.get_dir(), 'visualization.gif'), visualize_reset=False,
                      overlay_target=True, max_step=args.buffer_horizon, num_moves=args.num_moves)
    print("Mean performance so far:", calculate_performance(args.dataset_path, max_step=args.buffer_horizon, num_moves=args.num_moves))