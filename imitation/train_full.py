from imitation.agent import Agent
from imitation.sampler import sample_traj
from imitation.imitation_buffer import ImitationReplayBuffer, filter_buffer_nan
from imitation.utils import aggregate_traj_info
from tqdm import tqdm
import argparse
import random
import numpy as np
import torch
import json
import os
from chester import logger
#
from plb.envs.mp_wrapper import make_mp_envs
from imitation.utils import load_target_info
from imitation.utils import visualize_trajs
from imitation.eval_helper import eval_skills, eval_vae, eval_plan


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_args(cmd=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='Roll-v1')
    parser.add_argument('--num_env', type=int, default=1)  # Number of parallel environment
    parser.add_argument('--algo', type=str, default='imitation')
    parser.add_argument('--dataset_name', type=str, default='tmp')
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--gd_num_steps", type=int, default=50, help="steps for the gradient descent(gd) expert")

    # differentiable physics parameters
    parser.add_argument("--lr", type=float, default=0.02)  # For the solver
    parser.add_argument("--reset_lr", type=float, default=1e-3)
    parser.add_argument("--softness", type=float, default=666.)
    parser.add_argument("--optim", type=str, default='Adam', choices=['Adam', 'Momentum'])
    parser.add_argument("--num_trajs", type=int, default=20)  # Number of demonstration trajectories
    parser.add_argument("--energy_weight", type=float, default=0.)
    parser.add_argument("--vel_loss_weight", type=float, default=0.)

    # Train
    parser.add_argument("--il_num_epoch", type=int, default=5000)
    parser.add_argument("--il_lr", type=float, default=1e-3)
    parser.add_argument("--il_eval_freq", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--step_per_epoch", type=int, default=500)
    parser.add_argument("--step_warmup", type=int, default=2000)
    parser.add_argument("--hindsight_goal_ratio", type=float, default=0.5)
    parser.add_argument("--debug_overfit_test", type=bool, default=False)
    parser.add_argument("--obs_noise", type=float, default=0.05)
    parser.add_argument("--resume_path", default=None)
    parser.add_argument("--num_tools", type=int, default=2)
    parser.add_argument("--buffer_horizon", type=int, default=50)
    parser.add_argument("--gt_tool", type=bool, default=False)
    parser.add_argument("--open_loop", type=bool, default=False)

    # Architecture
    parser.add_argument("--frame_stack", type=int, default=1)
    parser.add_argument("--image_dim", type=int, default=64)
    parser.add_argument("--img_mode", type=str, default='rgb')
    parser.add_argument("--pos_ratio", type=float, default=0.5)
    parser.add_argument("--pos_reset_ratio", type=float, default=0.2)  # 20% of the positive goals will come from the reset motion
    parser.add_argument("--z_dim", type=int, default=32)  # Maybe try multiple values
    parser.add_argument("--actor_feature_dim", type=int, default=128)
    parser.add_argument("--encoder_beta", type=float, default=10.)
    parser.add_argument("--bin_succ", type=bool, default=False)

    # Plan
    parser.add_argument("--adam_sample", type=int, default=400)
    parser.add_argument("--adam_iter", type=int, default=3000)
    parser.add_argument("--adam_lr", type=float, default=5e-2)
    parser.add_argument("--min_zlogl", type=float, default=-30)
    parser.add_argument("--save_goal_his", type=bool, default=False)
    parser.add_argument("--plan_step", type=int, default=2)

    if cmd:
        args = parser.parse_args()
    else:
        args = parser.parse_args("")

    return args


def eval_training_traj(traj_ids, args, buffer, env, agent, target_imgs, np_target_imgs, np_target_mass_grids, save_name):
    horizon = 50
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


def prepare_agent_env(args):
    pass


def run_task(arg_vv, log_dir, exp_name):  # Chester launch
    args = get_args(cmd=False)

    args.__dict__.update(**arg_vv)

    set_random_seed(args.seed)

    # # Configure logger
    logger.configure(dir=log_dir, exp_name=exp_name)
    log_dir = logger.get_dir()
    assert log_dir is not None
    os.makedirs(log_dir, exist_ok=True)
    #
    # # Dump parameters
    with open(os.path.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2, sort_keys=True)

    # Need to make the environment before moving tensor to torch
    obs_channel = len(args.img_mode) * args.frame_stack
    img_obs_shape = (args.image_dim, args.image_dim, obs_channel)
    env = make_mp_envs(args.env_name, args.num_env, args.seed)

    args.cached_state_path = env.getattr('cfg.cached_state_path', 0)
    print(args.cached_state_path)
    action_dim = env.getattr('taichi_env.primitives.action_dim')[0]

    # Load buffer
    device = 'cuda'
    buffer = ImitationReplayBuffer(args)
    buffer.load(args.dataset_path)
    filter_buffer_nan(buffer)

    buffer.generate_train_eval_split()
    target_info = load_target_info(args, device)
    buffer.__dict__.update(**target_info)
    # torch.autograd.set_detect_anomaly(True)
    # # ----------preparation done------------------
    agent = Agent(args, None, img_obs_shape, action_dim, num_tools=2, device=device)
    if args.resume_path is not None:
        agent.load(args.resume_path)

    for epoch in range(args.il_num_epoch):
        epoch_tool_idxes = [buffer.get_epoch_tool_idx(epoch, tid) for tid in [0]]
        train_infos = []
        for batch_tools_idx in tqdm(zip(*epoch_tool_idxes)):
            data_batch = buffer.sample_tool_transitions(batch_tools_idx, epoch, device)
            train_info = agent.train(data_batch)
            train_infos.append(train_info)
        if epoch % args.il_eval_freq == 0:
            # Log training info
            train_infos = aggregate_traj_info(train_infos, prefix=None)

            # Evaluate skills
            skill_traj, skill_info = eval_skills(args, env, agent, epoch)
            vae_info = eval_vae(args, agent, skill_traj, epoch)

            # Plan
            if epoch % (args.il_eval_freq * 2) == 0:
                plan_info = eval_plan(args, env, agent, epoch)
            else:
                plan_info = {}

            # Logging
            logger.record_tabular('epoch', epoch)
            all_info = {}
            all_info.update(**train_infos)
            all_info.update(**skill_info)
            all_info.update(**vae_info)
            all_info.update(**plan_info)
            for key, val in all_info.items():
                logger.record_tabular(key, val)
            logger.dump_tabular()

            # Save model
            if epoch % (args.il_eval_freq * 2) == 0:
                agent.save(os.path.join(logger.get_dir(), f'agent_{epoch}.ckpt'))
    env.close()


if __name__ == '__main__':
    task = 'train_policy'
    if task == 'train_encoder':
        vv = {
            'task': 'train_encoder',
            'num_epoch': 500,
            'batch_size': 128,
            'dataset_path': 'data/autobot/0629_PushSpread/0629_PushSpread/0629_PushSpread_2021_06_29_23_32_22_0001/dataset.xz'
        }
        run_task(vv, './data/debug', 'test')
    elif task == 'train_policy':
        vv = {
            'task': 'train_policy',
            'il_eval_freq': 100,
            'num_epoch': 500,
            'batch_size': 128,
            'dataset_path': '/home/jianrenw/carl/research/dev/PlasticineLab/data/local/1020_Roll_exp_working/1020_Roll_exp_working_2021_10_21_11_26_07_0001/dataset.gz'
        }
        run_task(vv, './data/debug', 'test')
    elif task == 'train_feas':
        vv = {
            'task': 'train_feas',
            'il_eval_freq': 100,
            'num_epoch': 500,
            'batch_size': 128,
            'agent_dataset_path': None,
            'agent_path': 'data/local/0629_PushSpread_train_policy/0629_PushSpread_train_policy_2021_06_30_12_39_58_0001/agent_300.ckpt',
            'feas_num_traj': 10}
        run_task(vv, './data/debug-2', 'test')
    elif task == 'connect':
        vv = {
            'task': 'train_feas',
            'agent_path': 'data/local/0629_PushSpread_train_policy/0629_PushSpread_train_policy_2021_06_30_12_39_58_0001/agent_300.ckpt'}
        run_task(vv, './data/debug-2', 'test')
