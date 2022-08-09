import torch.multiprocessing as mp
import wandb

from imitation.utils import DARK_DOUGH, LIGHT_DOUGH, get_camera_matrix, get_partial_pcl2, load_target_info, write_number

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
import pickle
from chester import logger
from rl_sac.sac import SAC
from imitation.buffer import ReplayBuffer
from imitation.eval_helper import get_threshold


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(args, policy, eval_env, seed, tag, separate=False, dump_traj_dir=None):
    """ If dump_traj_dir is not None, then dump it and save the trajectory"""
    camera_view, camera_proj = get_camera_matrix(eval_env)
    episode_reward = []
    performance = []
    all_frames = []
    all_success = []
    all_trajs = []
    init_vs, target_vs = get_eval_traj(args.cached_state_path)
    n_eval = len(init_vs)
    while len(init_vs) % args.num_env != 0:
        init_vs.append(init_vs[-1])
        target_vs.append(target_vs[-1])

    for i in range(0, 10, args.num_env):
        state = eval_env.reset([{'init_v': init_vs[i + j], 'target_v': target_vs[i + j]} for j in range(args.num_env)])
        done = [False] * args.num_env
        obs = eval_env.render([{'mode':'rgb', 'img_size':128}]*args.num_env)
        target_imgs = eval_env.getattr('target_img')
        frames = [[obs[j][:, :, :3] * 0.8 + np.array(target_imgs[j])[:, :, :3] * 0.2] for j in range(args.num_env)]
        actions = [[] for _ in range(args.num_env)]
        rewards = [0. for _ in range(args.num_env)]
        while not done[0]:
            if args.use_pcl == 'partial_pcl':
                dough_pcl = [np.zeros((1000,3)) for _ in range(args.num_env)]
                tool_pcl = [np.zeros((100,3)) for _ in range(args.num_env)]
                goal_pcl = [np.zeros((1000, 3)) for _ in range(args.num_env)]
                dough_pcl_len, tool_pcl_len, goal_pcl_len = [], [], []
                for i in range(args.num_env):
                    pcl1 = get_partial_pcl2(obs[i], LIGHT_DOUGH, DARK_DOUGH, camera_view, camera_proj, random_mask=False, p=0.)[:,:3]
                    if len(pcl1) > 1000:
                        rand = np.random.choice(len(pcl1), size=1000, replace=False)
                        pcl1 = pcl1[rand]
                    pcl2 = state[i][3000:3300].reshape(-1, 3)
                    pcl = get_partial_pcl2(np.array(eval_env.getattr('target_img', i)), LIGHT_DOUGH, DARK_DOUGH, camera_view, camera_proj)[:, :3]
                    if len(pcl) > 1000:
                        rand = np.random.choice(len(pcl), size=1000, replace=False)
                        pcl = pcl[rand]

                    dough_pcl[i][:len(pcl1)] = pcl1
                    tool_pcl[i][:len(pcl2)] = pcl2
                    goal_pcl[i][:len(pcl)] = pcl

                    dough_pcl_len.append(len(pcl1))
                    tool_pcl_len.append(len(pcl2))
                    goal_pcl_len.append(len(pcl))
                action = list(policy.select_action_pcl(dough_pcl, dough_pcl_len, tool_pcl, tool_pcl_len, goal_pcl, goal_pcl_len, evaluate=True))
            else:
                action = list(policy.select_action(list(obs), target_imgs, evaluate=True))
            state, reward, done, infos = eval_env.step(action)
            obs = np.array(eval_env.render([{'mode':'rgb', 'img_size':128}]*args.num_env))
            for j in range(args.num_env):
                num = float(infos[j]['info_emd'])
                img = obs[j][:, :, :3] * 0.8 + target_imgs[j][:, :, :3] * 0.2
                write_number(img, num)
                frames[j].append(img)
                rewards[j] += reward[j]
                actions[j].append(action)
        for n in range(10):
            for j in range(args.num_env):
                if n == 0:
                    margin = 2
                    frames[j][-1][:margin, :] = frames[j][-1][-margin:, :] = frames[j][-1][:, :margin] = frames[j][-1][:, -margin:] = [0., 1., 0.]
                else:
                    frames[j].append(frames[j][-1])
        for j in range(args.num_env):
            merged_frames = []
            for t in range(len(frames[j])):
                merged_frames.append(frames[j][t])
            all_frames.append(merged_frames)
            performance.append(infos[j]['info_normalized_performance'])
            all_success.append(int(infos[j]['info_normalized_performance'] > get_threshold(args.env_name)))
            episode_reward.append(rewards[j])
            traj = {'init_v': init_vs[i + j], 
            'target_v': target_vs[i + j], 
            'actions': actions[j], 
            'info_normalized_performance': infos[j]['info_normalized_performance'],
            'episode_reward': rewards[j]}
            all_trajs.append(traj)
    all_frames, episode_reward, performance = all_frames[:n_eval], episode_reward[:n_eval], performance[:n_eval]
    if separate:
        for i, frames in enumerate(all_frames):
            gif_path = os.path.join(logger.get_dir(), f'eval_{tag}_{i}.gif')
            save_numpy_as_gif(np.array(frames), gif_path)
    else:
        all_frames = np.array(all_frames).swapaxes(0, 1)
        all_frames = [make_grid(all_frames[i], ncol=n_eval, padding=5, pad_value=0.5) for i in range(len(all_frames))]
        gif_path = os.path.join(logger.get_dir(), f'eval_{tag}.gif')
        save_numpy_as_gif(np.array(all_frames), gif_path)

    if dump_traj_dir is not None:
        for i in range(len(all_trajs)):
            with open(os.path.join(dump_traj_dir, f'execute_{i}.pkl'), 'wb') as f:
                pickle.dump(all_trajs[i], f)

    avg_reward = sum(episode_reward) / len(episode_reward)
    final_normalized_performance = np.mean(np.array(performance))
    logger.record_tabular('eval/episode_reward', avg_reward)
    logger.record_tabular('eval/final_normalized_performance', final_normalized_performance)
    logger.record_tabular('eval/success', np.array(all_success).mean())
    logger.info(str(all_success))
    return {'avg_reward': avg_reward, 'final_normalized_emd':final_normalized_performance}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="SAC")  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env_name", default='RollExp-v1', type=str)  # Environment name
    parser.add_argument("--num_env", default=1, type=int)  # Environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=2500, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=50, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=10000000, type=int)  # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=5, type=int)  # Batch size for both actor and critic
    parser.add_argument("--buffer_horizon", default=170, type=int)
    parser.add_argument("--gamma", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--resume_path", default=None)

    # Env
    parser.add_argument("--image_dim", type=int, default=64)
    parser.add_argument("--num_dough_points", type=int, default=1000)
    parser.add_argument("--num_tool_points", type=int, default=100)
    parser.add_argument("--img_mode", type=str, default='rgbd')
    parser.add_argument("--frame_stack", type=int, default=1)
    parser.add_argument("--tool_combo_id", type=int, default=0)

    # RL
    parser.add_argument("--replay_k", default=0, type=int, help='Number of imagined goals for each actual goal')
    parser.add_argument("--joint_opt", default=False, type=bool)
    parser.add_argument("--feature_dim", default=50, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--reward_type", default='emd', type=str)
    parser.add_argument("--emd_downsample_num", default=500, type=int)
    parser.add_argument("--use_pcl", type=str, default='partial_pcl')

    # SAC New
    parser.add_argument('--train_freq', default=170, type=int)
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G', help='Automaically adjust α (default: False)')
    parser.add_argument('--num_steps', type=int, default=1000001, metavar='N', help='maximum number of steps (default: 1000000)')
    parser.add_argument('--hidden_dim', type=int, default=256, metavar='N', help='hidden size (default: 256)')

    args, _ = parser.parse_known_args()
    return args


def train_sac(args, env):
    spec = get_tool_spec(env, args.env_name)
    args.action_mask = spec['action_masks'][args.tool_combo_id]
    args.contact_loss_mask = np.array(spec['contact_loss_masks'][args.tool_combo_id])
    args.contact_loss_mask_tensor = torch.FloatTensor(args.contact_loss_mask).to('cuda')
    args.discount = float(args.gamma)
    args.gt_tool = True

    if not args.use_pcl:
        obs_channel = len(args.img_mode) * args.frame_stack
        obs_shape = (args.image_dim, args.image_dim, obs_channel)
    else:
        obs_shape = (args.num_dough_points*2+ args.num_tool_points, 3*args.frame_stack)
    action_dim = env.getattr('action_space.shape[0]', 0)
    max_action = float(env.getattr('env.action_space.high[0]', 0))
    camera_view, camera_proj = get_camera_matrix(env)

    wandb.init(project=args.exp_prefix)
    wandb.config.update(args)
    if args.run_name != '':
        wandb.run.name = args.run_name
        wandb.run.save()
    
    # Agent
    kwargs = {
        'args': args,
        "obs_shape": obs_shape,
        "action_dim": action_dim,
        "max_action": max_action}

    policy = SAC(**kwargs)  # TODO

    if args.resume_path is not None:
        # TODO
        policy.load(args.resume_path)

    args.reward_fn = None
    state, done = env.reset([{'contact_loss_mask': args.contact_loss_mask}] * args.num_env), [False]
    obs = env.render([{'mode':'rgb', 'img_size': 128}] * args.num_env)

    replay_buffer = ReplayBuffer(args, maxlen=int(100000), her_args=args)
    device = 'cpu'
    target_info = load_target_info(args, device)
    replay_buffer.__dict__.update(**target_info)
    from rl_td3.run_td3 import Traj

    # Training Loop
    episode_timesteps = 0
    episode_num = 0

    curr_traj = Traj(args.num_env)

    for t in tqdm.trange(0, int(args.max_timesteps), args.num_env):
        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.getattr('action_space.sample()')
        else:
            if args.use_pcl == 'partial_pcl':
                action = policy.select_action_pcl(dough_pcl, dough_pcl_len, tool_pcl, tool_pcl_len, goal_pcl, goal_pcl_len)
            else:
                action = policy.select_action(obs, env.getattr('target_img'))  # Sample action from policy
            action = list(action)

        next_state, reward, done, info = env.step(action)
        next_obs = env.render([{'mode':'rgb', 'img_size': 128}] * args.num_env)

        episode_timesteps += args.num_env

        # Store data in replay buffer
        if args.use_pcl == 'partial_pcl':
            dough_pcl = [np.zeros((replay_buffer.max_pts['dough_pcl'],3)) for _ in range(args.num_env)]
            tool_pcl = [np.zeros((100,3)) for _ in range(args.num_env)]
            goal_pcl = [np.zeros((replay_buffer.max_pts['goal_pcl'], 3)) for _ in range(args.num_env)]
            dough_pcl_len, tool_pcl_len, goal_pcl_len = [], [], []
            for i in range(args.num_env):
                pcl1 = get_partial_pcl2(obs[i], LIGHT_DOUGH, DARK_DOUGH, camera_view, camera_proj, random_mask=False, p=0.)[:,:3]
                if len(pcl1) > replay_buffer.max_pts['dough_pcl']:
                    rand = np.random.choice(len(pcl1), size=replay_buffer.max_pts['dough_pcl'], replace=False)
                    pcl1 = pcl1[rand]
                pcl2 = state[i][3000:3300].reshape(-1, 3)
                pcl = get_partial_pcl2(np.array(env.getattr('target_img', i)), LIGHT_DOUGH, DARK_DOUGH, camera_view, camera_proj)[:, :3]
                if len(pcl) > replay_buffer.max_pts['goal_pcl']:
                    rand = np.random.choice(len(pcl), size=replay_buffer.max_pts['goal_pcl'], replace=False)
                    pcl = pcl[rand]

                dough_pcl[i][:len(pcl1)] = pcl1
                tool_pcl[i][:len(pcl2)] = pcl2
                goal_pcl[i][:len(pcl)] = pcl

                dough_pcl_len.append(len(pcl1))
                tool_pcl_len.append(len(pcl2))
                goal_pcl_len.append(len(pcl))


            curr_traj.add(states=state, actions=action, rewards=reward, init_v=env.getattr('init_v'), target_v=env.getattr('target_v'),
                      action_mask=args.action_mask, dough_pcl=dough_pcl, dough_pcl_len=dough_pcl_len, 
                      tool_pcl=tool_pcl, tool_pcl_len=tool_pcl_len, goal_pcl=goal_pcl, goal_pcl_len=goal_pcl_len)
        else:
            curr_traj.add(states=state, obses=obs, actions=action, rewards=reward, init_v=env.getattr('init_v'), target_v=env.getattr('target_v'),
                      action_mask=args.action_mask)
        obs = next_obs
        state = next_state
        
        all_info = {}
        # Train agent after collecting sufficient data
        if t >= args.start_timesteps and t % args.train_freq == 0:
            all_info = {'num_episode':episode_num,  'num_steps': t}
            qf1_losses, qf2_losses, policy_losses, alpha_losses, alpha_tlogses = [], [], [], [], []
            for _ in range(args.train_freq):
                qf1_loss, qf2_loss, policy_loss, alpha_loss, alpha_tlogs = policy.update_parameters(replay_buffer, args.batch_size)
                qf1_losses.append(qf1_loss)
                qf2_losses.append(qf2_loss)
                policy_losses.append(policy_loss)
                alpha_losses.append(alpha_loss)
                alpha_tlogses.append(alpha_tlogs)
            train_info = {'qf1_loss': np.mean(qf1_losses),
                            'qf2_loss': np.mean(qf2_losses),
                            'policy_loss': np.mean(policy_losses),
                            'alpha_loss': np.mean(alpha_losses),
                            'alpha_tlogs': np.mean(alpha_tlogs)}
                
            all_info.update(**train_info)
        if done[0]:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            # Add the final states and obs
            if args.use_pcl == 'partial_pcl':
                dough_pcl = [np.zeros((replay_buffer.max_pts['dough_pcl'],3)) for _ in range(args.num_env)]
                tool_pcl = [np.zeros((100,3)) for _ in range(args.num_env)]
                dough_pcl_len, tool_pcl_len= [], []
                for i in range(args.num_env):
                    pcl1 = get_partial_pcl2(obs[i], LIGHT_DOUGH, DARK_DOUGH, camera_view, camera_proj, random_mask=False, p=0.)[:,:3]
                    if len(pcl1) > replay_buffer.max_pts['dough_pcl']:
                        rand = np.random.choice(len(pcl1), size=replay_buffer.max_pts['dough_pcl'], replace=False)
                        pcl1 = pcl1[rand]
                    pcl2 = state[i][3000:3300].reshape(-1, 3)
    
                    dough_pcl[i][:len(pcl1)] = pcl1
                    tool_pcl[i][:len(pcl2)] = pcl2
                    dough_pcl_len.append(len(pcl1))
                    tool_pcl_len.append(len(pcl2))
                curr_traj.add(states=state, dough_pcl=dough_pcl, dough_pcl_len=dough_pcl_len, 
                        tool_pcl=tool_pcl, tool_pcl_len=tool_pcl_len, goal_pcl=goal_pcl, goal_pcl_len=goal_pcl_len)
            else:
                curr_traj.add(states=state, obses=obs)
            trajs = curr_traj.get_trajs()
            for traj in trajs:
                replay_buffer.add(traj)
            curr_traj = Traj(args.num_env)
            print(replay_buffer.cur_size)


            # Evaluate episode
            if episode_num % args.eval_freq == 0:
                # evaluations.append(
                eval_info = eval_policy(args, policy, env, args.seed, tag=episode_num)
                logger.record_tabular('num_episode', episode_num)
                # logger.record_tabular('num_step', episode_timesteps)
                logger.dump_tabular()
                if all_info == {}:
                    all_info = {'num_episode':episode_num,  'num_steps': t}
                all_info.update(**eval_info)
                policy.save(os.path.join(logger.get_dir(), 'model_{}'.format(episode_num)))

            # Reset environment
            state, done = env.reset([{'contact_loss_mask': args.contact_loss_mask}] * args.num_env), [False] * args.num_env
            obs = env.render([{'mode':'rgb', 'img_size':128}]*args.num_env)

            episode_timesteps = 0
            episode_num += args.num_env

        if all_info != {}:
            wandb.log(all_info)
    env.close()


def run_task(arg_vv, log_dir, exp_name):  # Chester launch
    from plb.envs.mp_wrapper import make_mp_envs
    args = get_args()
    args.__dict__.update(**arg_vv)

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
    train_sac(args, env)

# if __name__ == '__main__':
#     args_vv = {}
#     run_task(args_vv, 'debug_rl', 'rl')