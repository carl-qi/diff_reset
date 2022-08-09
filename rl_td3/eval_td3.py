import torch.multiprocessing as mp

try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass

import json
from chester import logger
import numpy as np
import torch
import argparse
import os
from rl_td3.td3 import TD3
from imitation.buffer import ReplayBuffer
from imitation.env_spec import get_tool_spec, get_threshold
from imitation.eval_helper import get_eval_traj
from plb.utils.visualization_utils import make_grid, save_numpy_as_gif
from plb.envs.mp_wrapper import make_mp_envs
from rl_td3.profiler import Profiler
import tqdm
from imitation.train_full import get_args


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(args, policy, eval_env, seed, tag):
    thr = get_threshold(args.env_name)
    episode_reward, performance, all_frames, successes = [], [], [], []
    init_vs, target_vs = get_eval_traj(args.cached_state_path)
    n_eval = len(init_vs)
    while len(init_vs) % args.num_env != 0:
        init_vs.append(init_vs[-1])
        target_vs.append(target_vs[-1])

    for i in range(0, len(init_vs), args.num_env):
        _ = eval_env.reset([{'init_v': init_vs[i + j], 'target_v': target_vs[i + j]} for j in range(args.num_env)])
        done = [False] * args.num_env
        obs = eval_env.render(mode='rgb')
        target_imgs = eval_env.getattr('target_img')
        frames = [[obs[j][:, :, :3] * 0.8 + np.array(target_imgs[j])[:, :, :3] * 0.2] for j in range(args.num_env)]
        rewards = [0. for _ in range(args.num_env)]
        while not done[0]:
            action = list(policy.select_action(list(obs), target_imgs))
            state, reward, done, infos = eval_env.step(action)
            obs = np.array(eval_env.render(mode='rgb'))
            for j in range(args.num_env):
                frames[j].append(obs[j][:, :, :3] * 0.8 + target_imgs[j][:, :, :3] * 0.2)
                rewards[j] += reward[j]

        for j in range(args.num_env):
            merged_frames = []
            for t in range(len(frames[j])):
                merged_frames.append(frames[j][t])
            all_frames.append(merged_frames)
            performance.append(infos[j]['info_normalized_performance'])
            successes.append(int(infos[j]['info_normalized_performance'] > thr))
            episode_reward.append(rewards[j])
    all_frames, episode_reward, performance = all_frames[:n_eval], episode_reward[:n_eval], performance[:n_eval]
    all_frames = np.array(all_frames).swapaxes(0, 1)
    all_frames = [make_grid(all_frames[i], ncol=n_eval, padding=5, pad_value=0.5) for i in range(len(all_frames))]
    gif_path = os.path.join(logger.get_dir(), f'eval_{tag}.gif')
    save_numpy_as_gif(np.array(all_frames), gif_path)

    avg_reward = sum(episode_reward) / len(episode_reward)
    final_normalized_performance = np.mean(np.array(performance))
    print('successes:', successes)
    logger.record_tabular('eval/episode_reward', avg_reward)
    logger.record_tabular('eval/final_normalized_performance', final_normalized_performance)
    logger.record_tabular('eval/success', np.mean(successes))


def run_task(arg_vv, log_dir, exp_name):  # Chester launch
    exp_folder = arg_vv['exp_folder']
    with open(os.path.join(exp_folder, 'variant.json'), 'rb') as f:
        vv = json.load(f)
    args = get_args()
    args.__dict__.update(**vv)
    args.num_env = 1
    tag = args.env_name + '_' + str(args.tool_combo_id) + '_' + str(args.seed)

    # Configure logger
    logger.configure(dir=os.path.join('./data/eval_rl/', tag), exp_name='')
    log_dir = logger.get_dir()
    assert log_dir is not None
    os.makedirs(log_dir, exist_ok=True)
    print("number of devices in current env ", torch.cuda.device_count())

    with open(os.path.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2, sort_keys=True)

    # ----------preparation done------------------
    env = make_mp_envs(args.env_name, args.num_env, args.seed)
    args.cached_state_path = env.getattr('cfg.cached_state_path', 0)

    spec = get_tool_spec(env, args.env_name)
    args.action_mask = spec['action_masks'][args.tool_combo_id]
    args.discount = float(args.gamma)

    obs_channel = len(args.img_mode) * args.frame_stack
    obs_shape = (args.image_dim, args.image_dim, obs_channel)
    action_dim = env.getattr('action_space.shape[0]', 0)
    max_action = float(env.getattr('env.action_space.high[0]', 0))

    kwargs = {
        'args': args,
        "obs_shape": obs_shape,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau}

    # Initialize policy
    if args.policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = TD3(**kwargs)
    else:
        raise NotImplementedError

    import glob
    filename = glob.glob(os.path.join(exp_folder, 'model_*_actor.pth'))[0]
    filename = filename[:-10]
    policy.load(filename)
    eval_policy(args, policy, env, args.seed, tag)
    logger.dump_tabular()


if __name__ == '__main__':
    exp_folders = [
        'data/hza/rl/gathermove-0-100/0928_td3/0928_td3_2021_10_01_06_11_21_0001/',
        'data/hza/rl/gathermove-0-200/0928_td3/0928_td3_2021_10_01_06_11_22_0001/',
        'data/hza/rl/gathermove-1-100/0928_td3/0928_td3_2021_10_01_06_11_24_0001/',
        'data/hza/rl/gathermove-1-200/0928_td3/0928_td3_2021_10_01_06_17_30_0001/',
        'data/hza/rl/gathermove-2-100/0928_td3/0928_td3_2021_10_01_21_09_52_0001/',
        'data/hza/rl/gathermove-2-200/0928_td3/0928_td3_2021_10_02_03_14_32_0001/']
    for exp_folder in exp_folders:
        run_task(exp_folder)
