from imitation.env_spec import set_render_mode
from plb.algorithms.bc.bc_agent import Agent
from plb.algorithms.bc.ibc_agent import Agent as IBCAgent
from imitation.sampler import sample_traj
from imitation.imitation_buffer import ImitationReplayBuffer, filter_buffer_nan, filter_hard_coded_actions, segment_partial_pcl
from imitation.utils import aggregate_traj_info, get_camera_matrix
from tqdm import tqdm
import argparse
import random
import numpy as np
import torch

from plb.envs.multitask_env import MultitaskPlasticineEnv
torch.multiprocessing.set_start_method('spawn')# good solution !!!!
# torch.multiprocessing.set_start_method('forkserver', force=True)
import json
import os
from chester import logger
from plb.envs.mp_wrapper import make_mp_envs
from imitation.utils import load_target_info
from imitation.utils import visualize_trajs, LIGHT_DOUGH, LIGHT_TOOL, DARK_DOUGH, DARK_TOOL
import wandb
from imitation.args import get_args


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def eval_traj(traj_ids, args, buffer, env, agent, np_target_imgs, save_name, visualize=False):
    horizon = buffer.horizon
    trajs = []
    demo_obses = []
    agent.actors[0].eval()
    for traj_id in traj_ids:
        init_v = int(buffer.buffer['init_v'][traj_id * horizon])
        target_v = int(buffer.buffer['target_v'][traj_id * horizon])
        reset_key = {'init_v': init_v, 'target_v': target_v}
        tid = buffer.get_tid(buffer.buffer['action_mask'][traj_id * horizon])
        traj = sample_traj(env, agent, reset_key, tid, buffer=buffer)
        traj['target_img'] = np_target_imgs[reset_key['target_v']]
        demo_obs = buffer.buffer['obses'][traj_id * args.buffer_horizon: (traj_id+1) * args.buffer_horizon]
        demo_obses.append(demo_obs)
        # demo_target_ious.append(buffer.buffer['target_ious'][traj_id * horizon + horizon - 1])
        print(f'tid: {tid}, traj_id: {traj_id}, reward: {np.sum(traj["rewards"])}')
        trajs.append(traj)
    demo_obses = np.array(demo_obses)

    # agent_ious = np.array([traj['target_ious'][-1, 0] for traj in trajs])
    # demo_target_ious = np.array(demo_target_ious)
    # logger.log('Agent ious: {}, Demo ious: {}'.format(np.mean(agent_ious), np.mean(demo_target_ious)))
    if visualize:
        visualize_trajs(trajs, 5, key='info_emds', save_name=os.path.join(logger.get_dir(), save_name),
                        vis_target=True, demo_obses=demo_obses[:, :, :, :, :3])
    # info = {'agent_iou': np.mean(agent_ious), 'demo_iou': np.mean(demo_target_ious)}
    final_perf_normalized = np.mean([t['info_final_normalized_performance'] for t in trajs])
    # avg_perf_normalized = np.mean([np.mean(t['info_normalized_performance']) for t in trajs])
    final_perf_normalized_std = np.std([t['info_final_normalized_performance'] for t in trajs])
    info = {'eval_final_normalized_performance': final_perf_normalized,
    # 'eval_avg_normalized_performance': avg_perf_normalized,
    'eval_final_normalized_performance_std': final_perf_normalized_std}
    return info


def prepare_buffer(args, camera_view, camera_proj):
    buffer = ImitationReplayBuffer(args)
    buffer.load(args.dataset_path)
    filter_buffer_nan(buffer)
    print("buffer size:", buffer.cur_size)
    if buffer.horizon == 170:
        filter_hard_coded_actions(buffer, 50, 120)
        buffer.horizon = 100
    # buffer.generate_train_eval_split(traj_limit=args.traj_limit, eval_train=args.eval_train)
    buffer.generate_train_eval_split2(eval_train=args.eval_train)
    target_info = load_target_info(args, 'cuda')
    buffer.__dict__.update(**target_info)
    if args.use_pcl == 'partial_pcl':
        segment_partial_pcl(buffer, 'tool_pcl', LIGHT_TOOL, DARK_TOOL, camera_view, camera_proj)
        segment_partial_pcl(buffer, 'dough_pcl', LIGHT_DOUGH, DARK_DOUGH, camera_view, camera_proj)
        segment_partial_pcl(buffer, 'goal_pcl', LIGHT_DOUGH, DARK_DOUGH, camera_view, camera_proj)
    if True:
        buffer.compute_stats(buffer.train_idx, 'cuda')
    return buffer

def run_task(arg_vv, log_dir, exp_name):  # Chester launch
    args = get_args(cmd=False)

    args.__dict__.update(**arg_vv)

    set_random_seed(args.seed)

    # # Configure logger
    logger.configure(dir=log_dir, exp_name=exp_name)
    log_dir = logger.get_dir()
    assert log_dir is not None
    os.makedirs(log_dir, exist_ok=True)

    wandb.init(project=args.exp_prefix)
    wandb.config.update(args)
    wandb.run.name = "PN++_stack{0}_{1}traj_lr{2}".format(args.frame_stack, args.traj_limit, args.il_lr) if args.run_name == '' else args.run_name
    wandb.run.save()

    # # Dump parameters
    with open(os.path.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2, sort_keys=True)

    # Need to make the environment before moving tensor to torch
    obs_channel = len(args.img_mode) * args.frame_stack
    img_obs_shape = (args.image_dim, args.image_dim, obs_channel)

    point_channel = 3 * args.frame_stack
    point_obs_shape = (1100, point_channel)
    obs_shape = img_obs_shape if not args.use_pcl else point_obs_shape

    env = make_mp_envs(args.env_name, args.num_env, args.seed)

    args.cached_state_path = env.getattr('cfg.cached_state_path', 0)
    print(args.cached_state_path)
    action_dim = env.getattr('taichi_env.primitives.action_dim')[0]
    cfg_path = env.getattr('cfg_path', 0)
    camera_view, camera_proj = get_camera_matrix(env)
    
    # Load buffer
    device = 'cuda'
    buffer = prepare_buffer(args, camera_view, camera_proj)
    # torch.autograd.set_detect_anomaly(True)
    # # ----------preparation done------------------
    if args.open_loop:
        action_dim = action_dim * buffer.horizon
        agent = Agent(args, None, obs_shape, action_dim, num_tools=1, device=device) # TODO: write pcl agent
        if args.resume_path is not None:
            agent.load(args.resume_path)
        total_steps = 0
        for epoch in range(args.il_num_epoch):
            print('----------- epoch: {0} -----------'.format(epoch))
            train_infos = []
            traj_idxes = np.random.permutation(buffer.train_traj_idx)
            n = len(traj_idxes) // args.batch_size
            for i in tqdm(range(n)):
                batch_traj_idxes = traj_idxes[i*args.batch_size:(i+1)*args.batch_size]
                data_batch = buffer.sample_transition_openloop(batch_traj_idxes, device)
                train_info = agent.train(data_batch, stats=buffer.stats, pcl=args.use_pcl)

            total_steps += len(traj_idxes)
            train_infos.append(train_info)
            del data_batch

            if epoch % args.il_eval_freq == 0:
                # Log training info
                train_infos = aggregate_traj_info(train_infos, prefix=None)

                # evaluate
                if epoch % (args.il_eval_freq) == 0:
                    eval_infos = []
                    num_batch = len(buffer.eval_traj_idx) // args.batch_size
                    for  i in range(num_batch):
                        data_batch = buffer.sample_transition_openloop(buffer.eval_traj_idx[i*args.batch_size:(i+1)*args.batch_size], device)
                        eval_infos.append(agent.eval(data_batch, pcl=args.use_pcl))
                    del data_batch
                    eval_infos = aggregate_traj_info(eval_infos, prefix=None)
                plan_info = eval_traj(buffer.hard_coded_eval_idxes[:10], args, buffer, env, agent, buffer.np_target_imgs, f'visual_{epoch}.gif',visualize=True)
                # Logging
                logger.record_tabular('epoch', epoch)
                logger.record_tabular('total steps', total_steps)
                all_info = {}
                all_info.update(**train_infos)
                all_info.update(**eval_infos)
                all_info.update(**plan_info)
                all_info.update({'epoch': epoch, 'total steps': total_steps})
                wandb.log(all_info)
                for key, val in all_info.items():
                    logger.record_tabular(key, val)
                logger.dump_tabular()

                # Save model
                if epoch % (args.il_eval_freq) == 0:
                    agent.save(os.path.join(logger.get_dir(), f'agent_{epoch}.ckpt'))
        env.close()
        exit(0)


    # closed loop
    agent = Agent(args, None, obs_shape, action_dim, num_tools=1, device=device) # TODO: write pcl agent
    if args.resume_path is not None:
        agent.load(args.resume_path)

    total_steps = 0
    for epoch in range(args.il_num_epoch):
        print('----------- epoch: {0} -----------'.format(epoch))
        # import pdb; pdb.set_trace()
        epoch_tool_idxes = [buffer.get_epoch_tool_idx(epoch, tid) for tid in [0]]
        train_infos = []
        for batch_tools_idx in tqdm(zip(*epoch_tool_idxes)):
            data_batch = buffer.sample_tool_transitions_bc(batch_tools_idx, epoch, device, pcl=args.use_pcl)
            if args.actor_type == 'PointActorToolParticle':
                train_info = agent.train_tool(data_batch, stats=buffer.stats, pcl=args.use_pcl)
            else:
                train_info = agent.train(data_batch, stats=buffer.stats, pcl=args.use_pcl)
            total_steps += len(batch_tools_idx) * batch_tools_idx[0].shape[0]
            train_infos.append(train_info)
        del data_batch
        if isinstance(agent, IBCAgent) and epoch % 100 == 0:
            agent.scheduler.step()
        if epoch % args.il_eval_freq == 0:
            # Log training info
            train_infos = aggregate_traj_info(train_infos, prefix=None)

            # evaluate
            if epoch % (args.il_eval_freq) == 0:
                eval_infos = []
                num_batch = len(buffer.eval_idx) // args.batch_size
                for  i in range(num_batch):
                    data_batch = buffer.sample_tool_transitions_bc([buffer.eval_idx[i*args.batch_size:(i+1)*args.batch_size],], \
                                                                    epoch, device, pcl=args.use_pcl)
                    eval_infos.append(agent.eval(data_batch, pcl=args.use_pcl))
                del data_batch
                eval_infos = aggregate_traj_info(eval_infos, prefix=None)
            if True:
                plan_info = eval_traj(buffer.hard_coded_eval_idxes[:10], args, buffer, env, agent, buffer.np_target_imgs, f'visual_{epoch}.gif',visualize=True)
            else:
                plan_info = {}
            # Logging
            logger.record_tabular('epoch', epoch)
            logger.record_tabular('total steps', total_steps)
            all_info = {}
            all_info.update(**train_infos)
            all_info.update(**eval_infos)
            all_info.update(**plan_info)
            all_info.update({'epoch': epoch, 'total steps': total_steps})
            wandb.log(all_info)
            for key, val in all_info.items():
                logger.record_tabular(key, val)
            logger.dump_tabular()

            # Save model
            if epoch % (args.il_eval_freq) == 0:
                agent.save(os.path.join(logger.get_dir(), f'agent_{epoch}.ckpt'))
    env.close()


# if __name__ == '__main__':

#     vv = {
#         'task': 'train_policy',
#         'il_eval_freq': 1,
#         'frame_stack': 1,
#         'use_pcl': 'full_pcl',
#         'il_num_epoch': 500,
#         'il_eval_freq':20,
#         'batch_size': 10,
#         "step_per_epoch": 0,
#         "run_name": "",
#         "hindsight_goal_ratio": 0.,
#         "ir_lr": 0.0001,
#         "obs_noise": 0.,
#         "buffer_horizon": 170,
#         'traj_limit': 1,
#         'eval_train':True,
#         'dataset_path': '/home/jianrenw/carl/research/dev/PlasticineLab/data/seuss/1129_Roll_exp_gendemo_new/1129_Roll_exp_gendemo_new/1129_Roll_exp_gendemo_new_2021_12_05_15_37_13_0001/dataset.gz'
#     }
#     run_task(vv, './data/debug_bc', 'test')
