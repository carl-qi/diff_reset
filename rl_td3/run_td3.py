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
from imitation.env_spec import get_tool_spec
from imitation.env_spec import get_eval_traj
from plb.utils.visualization_utils import make_grid, save_numpy_as_gif
from plb.envs.mp_wrapper import make_mp_envs
from rl_td3.profiler import Profiler
import tqdm


class Traj(object):
    def __init__(self, num_env):
        self.num_env = num_env
        self.traj = [{} for _ in range(num_env)]
        self.keys = ['states', 'actions', 'rewards', 'init_v', 'target_v', 'action_mask', 'dough_pcl', 
        'dough_pcl_len', 'goal_pcl', 'goal_pcl_len', 'tool_pcl', 'tool_pcl_len']
        for i in range(num_env):
            for key in self.keys:
                self.traj[i][key] = []

    def add(self, **kwargs):
        for i in range(self.num_env):
            for key, val in kwargs.items():
                self.traj[i][key].append(val[i])

    def get_trajs(self):
        trajs = []
        for i in range(self.num_env):
            traj = {}
            for key in self.keys:
                traj[key] = np.array(self.traj[i][key])
            trajs.append(traj)
        return trajs


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env_name", default='PushSpread-v1', type=str)  # Environment name
    parser.add_argument("--num_env", default=1, type=int)  # Environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=2500, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=200, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=500000, type=int)  # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=128, type=int)  # Batch size for both actor and critic
    parser.add_argument("--gamma", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name

    # Env
    parser.add_argument("--image_dim", type=int, default=64)
    parser.add_argument("--img_mode", type=str, default='rgbd')
    parser.add_argument("--frame_stack", type=int, default=1)

    # RL
    parser.add_argument("--replay_k", default=4, type=int, help='Number of imagined goals for each actual goal')
    parser.add_argument("--joint_opt", default=False, type=bool)
    parser.add_argument("--feature_dim", default=50, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--reward_type", default='emd', type=str)
    parser.add_argument("--emd_downsample_num", default=1000, type=int)

    # Hindsight experience replay

    args, _ = parser.parse_known_args()
    return args


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(args, policy, eval_env, seed, tag):
    episode_reward = []
    performance = []
    all_frames = []
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
            episode_reward.append(rewards[j])
    all_frames, episode_reward, performance = all_frames[:n_eval], episode_reward[:n_eval], performance[:n_eval]
    all_frames = np.array(all_frames).swapaxes(0, 1)
    all_frames = [make_grid(all_frames[i], ncol=n_eval, padding=5, pad_value=0.5) for i in range(len(all_frames))]
    gif_path = os.path.join(logger.get_dir(), f'eval_{tag}.gif')
    save_numpy_as_gif(np.array(all_frames), gif_path)

    avg_reward = sum(episode_reward) / len(episode_reward)
    final_normalized_performance = np.mean(np.array(performance))
    logger.record_tabular('eval/episode_reward', avg_reward)
    logger.record_tabular('eval/final_normalized_performance', final_normalized_performance)


def train_td3(args, env):
    spec = get_tool_spec(env, args.env_name)
    args.action_mask = spec['action_masks'][args.tool_combo_id]
    args.contact_loss_mask = np.array(spec['contact_loss_masks'][args.tool_combo_id])
    args.contact_loss_mask_tensor = torch.FloatTensor(args.contact_loss_mask).to('cuda')
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

    if args.resume_path is not None:
        # TODO also find the progress.csv and load that
        policy.load(args.resume_path)

    state, done = env.reset([{'contact_loss_mask': args.contact_loss_mask}] * args.num_env), [False]

    def make_her_reward_fn(args, env):
        if args.reward_type == 'emd':

            downsample_num = args.emd_downsample_num

            from geomloss import SamplesLoss
            loss_fn = SamplesLoss(loss='sinkhorn', p=1, blur=0.001)

            def reward(tool_state, particle_state, goal_particle_state):
                if not isinstance(tool_state, torch.Tensor):
                    tool_state = torch.FloatTensor(tool_state).to('cuda', non_blocking=True).contiguous()
                    particle_state = torch.FloatTensor(particle_state).to('cuda', non_blocking=True).contiguous()
                    goal_particle_state = torch.FloatTensor(goal_particle_state).to('cuda', non_blocking=True).contiguous()
                emd = loss_fn(particle_state, goal_particle_state)
                dists = torch.min(torch.cdist(tool_state.transpose(1, 2)[:, :3], particle_state), dim=2)[0]
                contact_loss = (args.contact_loss_mask_tensor[None] * dists).sum(dim=1) / dists.shape[0] * 1e-3
                reward = -emd - contact_loss
                return reward.detach().cpu().numpy()

            def state_to_dict(vec):
                # TODO Remove the hard coding here
                if len(vec.shape) == 1:
                    particles = vec[:1000 * 3].reshape([1000, 3])
                    tool_state = vec[1000 * 3:].reshape([3, 8])
                elif len(vec.shape) == 2:
                    N = vec.shape[0]
                    particles = vec[:, :1000 * 3].reshape([N, 1000, 3])
                    tool_state = vec[:, 1000 * 3:].reshape([N, 3, 8])
                return {'particles': particles, 'tool_state': tool_state}

            def her_reward_fn(curr_state, goal_state):
                curr_d, goal_d = state_to_dict(curr_state), state_to_dict(goal_state)
                curr_p, curr_tool_state, goal_p = curr_d['particles'], curr_d['tool_state'], goal_d['particles'],
                rs = []
                N = curr_p.shape[1]
                sample_idx = np.random.choice(range(N), size=downsample_num, replace=False)
                B = 128
                for i in range(0, len(curr_p), B):
                    with torch.no_grad():
                        r = reward(curr_tool_state[i:i + B], curr_p[i:i + B, sample_idx], goal_p[i:i + B, sample_idx])
                    rs.append(r)
                return np.concatenate(rs)
        elif args.reward_type == 'chamfer':
            from imitation.utils import chamfer_distance
            raise NotImplementedError

        else:
            raise NotImplementedError

        return her_reward_fn

    args.reward_fn = make_her_reward_fn(args, env)
    replay_buffer = ReplayBuffer(args, her_args=args)

    obs = env.render(mode='rgb')
    episode_timesteps = 0
    episode_num = 0

    curr_traj = Traj(args.num_env)

    if args.profile:
        pr = Profiler()
    for t in tqdm.trange(0, int(args.max_timesteps), args.num_env):
        if args.profile and t % 50 == 0:
            pr.start_profile()
        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.getattr('action_space.sample()')
        else:
            action = (
              policy.select_action(obs, env.getattr('target_img'))
              + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)
            action = list(action)

        # Perform action
        next_state, reward, done, info = env.step(action)
        next_obs = env.render(mode='rgb')

        episode_timesteps += args.num_env

        # Store data in replay buffer
        curr_traj.add(states=state, obses=obs, actions=action, rewards=reward, init_v=env.getattr('init_v'), target_v=env.getattr('target_v'),
                      action_mask=args.action_mask)
        obs = next_obs
        state = next_state

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps and t % args.train_freq == 0:
            for _ in range(args.train_freq):
                policy.train(replay_buffer, args.batch_size)

        if done[0]:  # TODO If any env is done. This is fine for now since they are all 50
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            curr_traj.add(states=state, obses=obs)  # Add the final states and obs
            trajs = curr_traj.get_trajs()
            for traj in trajs:
                replay_buffer.add(traj)
            curr_traj = Traj(args.num_env)

            # Reset environment
            state, done = env.reset([{'contact_loss_mask': args.contact_loss_mask}] * args.num_env), [False] * args.num_env
            obs = env.render(mode='rgb')

            episode_timesteps = 0
            episode_num += args.num_env

            # Evaluate episode
            if episode_num % args.eval_freq == 0:
                # evaluations.append(
                eval_policy(args, policy, env, args.seed, tag=episode_num)
                logger.record_tabular('num_episode', episode_num)
                # logger.record_tabular('num_step', episode_timesteps)
                logger.dump_tabular()

                policy.save(os.path.join(logger.get_dir(), 'model_{}'.format(episode_num)))

        if args.profile and t % 50 == 49:
            pr.end_profile()


def run_task(arg_vv, log_dir, exp_name):  # Chester launch
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
    train_td3(args, env)
