from cProfile import run
import os
import cv2
import wandb
from chester import logger
import json
import random
import torch
import pickle
import argparse
from imitation.env_spec import get_threshold, get_tool_spec, set_render_mode
from imitation.hardcoded_eval_trajs import get_eval_traj
from plb.envs import make
import numpy as np
import scipy.stats as stats
from tqdm import tqdm
from plb.envs.mp_wrapper import SubprocVecEnv, make_mp_envs
from plb.utils.visualization_utils import make_grid, save_numpy_as_gif


def get_args(cmd=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--replay", action="store_true")
    parser.add_argument("-mpc", "--use_mpc", action="store_true")
    parser.add_argument("--env_name", default="RollExp-v4")
    parser.add_argument('--exp_prefix', type=str, default='')
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--plan_horizon", type=int, default=5)
    parser.add_argument("--max_iters", type=int, default=5)
    parser.add_argument("--population_size", type=int, default=50)
    parser.add_argument("--num_elites", type=int, default=10)
    if cmd:
        args = parser.parse_args()
    else:
        args = parser.parse_args("")

    return args

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def eval_policy(args, policy, eval_env, seed, tag, separate=False, dump_traj_dir=None):
    """ If dump_traj_dir is not None, then dump it and save the trajectory"""
    episode_reward = []
    performance = []
    all_frames = []
    # all_success = []
    all_trajs = []
    init_vs, target_vs = get_eval_traj(eval_env.cfg.cached_state_path)
    n_eval = len(init_vs[:10])
    for i in range(0, n_eval):
        tool_spec = get_tool_spec(eval_env, args.env_name)
        action_mask = tool_spec['action_masks'][0]
        contact_loss_mask = tool_spec['contact_loss_masks'][0]
        eval_env.reset(init_v=init_vs[i], target_v=target_vs[i], contact_loss_mask=contact_loss_mask)

        obs = eval_env.render(mode='rgb', img_size=128)
        target_imgs = eval_env.target_img
        frames = [obs[:, :, :3] * 0.8 + np.array(target_imgs)[:, :, :3] * 0.2]
        actions = []
        rewards = 0.
        T = eval_env._max_episode_steps
        for t in range(T):
            print("traj {}, timestep {}".format(i, t))
            action = policy.get_action()
            state, reward, done, infos = eval_env.step(action)
            obs = np.array(eval_env.render(mode='rgb', img_size=128))
                
            frames.append(obs[:, :, :3] * 0.8 + target_imgs[:, :, :3] * 0.2)
            rewards += reward
            actions.append(action)

        merged_frames = []
        for t in range(len(frames)):
            merged_frames.append(frames[t])
        all_frames.append(merged_frames)
        performance.append(infos['info_normalized_performance'])
        # all_success.append(int(infos['info_normalized_performance'] > get_threshold(args.env_name)))
        episode_reward.append(rewards)
        traj = {'init_v': init_vs[i], 
        'target_v': target_vs[i], 
        'actions': actions, 
        'info_normalized_performance': infos['info_normalized_performance'],
        'episode_rewaard': rewards}
        all_trajs.append(traj)
        wandb.log({'episode_reward': rewards, 'final_normalized_emd':infos['info_normalized_performance']})
        
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
    # logger.record_tabular('eval/success', np.array(all_success).mean())
    # logger.info(str(all_success))
    eval_info = {'avg_reward': avg_reward, 'final_normalized_emd':final_normalized_performance}
    return eval_info


class CEMOptimizer(object):
    def __init__(self, cost_function, solution_dim, max_iters, population_size, num_elites,
                 upper_bound=None, lower_bound=None, epsilon=0.05):
        """
        :param cost_function: Takes input one or multiple data points in R^{sol_dim}\
        :param solution_dim: The dimensionality of the problem space
        :param max_iters: The maximum number of iterations to perform during optimization
        :param population_size: The number of candidate solutions to be sampled at every iteration
        :param num_elites: The number of top solutions that will be used to obtain the distribution
                            at the next iteration.
        :param upper_bound: An array of upper bounds for the sampled data points
        :param lower_bound: An array of lower bounds for the sampled data points
        :param epsilon: A minimum variance. If the maximum variance drops below epsilon, optimization is stopped.
        """
        super().__init__()
        self.solution_dim, self.max_iters, self.population_size, self.num_elites = \
            solution_dim, max_iters, population_size, num_elites

        self.ub, self.lb = upper_bound.reshape([1, solution_dim]), lower_bound.reshape([1, solution_dim])
        self.epsilon = epsilon

        if num_elites > population_size:
            raise ValueError("Number of elites must be at most the population size.")

        self.cost_function = cost_function

    def obtain_solution(self, cur_state, init_mean=None, init_var=None):
        """ Optimizes the cost function using the provided initial candidate distribution
        :param cur_state: Full state of the current environment such that the environment can always be reset to this state
        :param init_mean: (np.ndarray) The mean of the initial candidate distribution.
        :param init_var: (np.ndarray) The variance of the initial candidate distribution.
        :return:
        """
        mean = (self.ub + self.lb) / 2. if init_mean is None else init_mean
        var = (self.ub - self.lb) / 4. if init_var is None else init_var
        t = 0
        X = stats.norm(loc=np.zeros_like(mean), scale=np.ones_like(mean))

        while (t < self.max_iters):  # and np.max(var) > self.epsilon:
            print("inside CEM, iteration {}".format(t))
            samples = X.rvs(size=[self.population_size, self.solution_dim]) * np.sqrt(var) + mean
            samples = np.clip(samples, self.lb, self.ub)
            costs = self.cost_function(cur_state, samples)
            sort_costs = np.argsort(costs)

            elites = samples[sort_costs][:self.num_elites]
            mean = np.mean(elites, axis=0)
            var = np.var(elites, axis=0)
            t += 1
        sol, solvar = mean, var
        return sol


class CEMPolicy(object):
    """ Use the ground truth dynamics to optimize a trajectory of actions. """

    def __init__(self, env, use_mpc, plan_horizon, max_iters, population_size, num_elites):
        self.env = env
        self.use_mpc = use_mpc
        self.plan_horizon, self.action_dim = plan_horizon, len(env.action_space.sample())
        self.action_buffer = []
        self.prev_sol = None

        lower_bound = np.tile(env.action_space.low[None], [self.plan_horizon, 1]).flatten()
        upper_bound = np.tile(env.action_space.high[None], [self.plan_horizon, 1]).flatten()
        self.optimizer = CEMOptimizer(self.cost_function,
                                      self.plan_horizon * self.action_dim,
                                      max_iters=max_iters,
                                      population_size=population_size,
                                      num_elites=num_elites,
                                      lower_bound=lower_bound,
                                      upper_bound=upper_bound)

    def cost_function(self, cur_state, action_trajs):
        env = self.env
        env.set_state(cur_state)
        action_trajs = action_trajs.reshape([-1, self.plan_horizon, self.action_dim])
        n = action_trajs.shape[0]
        costs = []
        print('evalute trajectories...')
        for i in tqdm(range(n)):
            env.set_state(cur_state)
            ret = 0
            for j in range(self.plan_horizon):
                _, reward, _, _ = env.step(action_trajs[i, j, :])
                ret += reward
            costs.append(-ret)
        return costs

    def reset(self):
        self.prev_sol = None

    def get_action(self):
        if len(self.action_buffer) > 0 and self.use_mpc:
            action, self.action_buffer = self.action_buffer[0], self.action_buffer[1:]
            return action
        self.env.debug = False
        # env_state = self.env.getfunc('get_state')[0]
        env_state = self.env.get_state()

        soln = self.optimizer.obtain_solution(env_state, self.prev_sol).reshape([-1, self.action_dim])
        if self.use_mpc:
            self.prev_sol = np.vstack([np.copy(soln)[1:, :], np.zeros([1, self.action_dim])]).flatten()
        else:
            self.prev_sol = None
            self.action_buffer = soln[1:]  # self.action_buffer is only needed for the non-mpc case.
        self.env.set_state(env_state)  # Recover the environment
        print("cem finished planning!")
        return soln[0]


def solve_cem(env, args):
    # T = env.getattr('_max_episode_steps', 0)
    T = env._max_episode_steps
    if args.use_mpc:
        args.population_size = args.population_size // args.plan_horizon

    print('env_horizon:', T)
    print('plan_horizon:', args.plan_horizon)
    print('population_size:', args.population_size)
    print('num_elites:', args.num_elites)
    policy = CEMPolicy(env,
                       args.use_mpc,
                       plan_horizon=args.plan_horizon,
                       max_iters=args.max_iters,
                       population_size=args.population_size,
                       num_elites=args.num_elites)
    # Run policy
    # eval_env = make(args.env_name)
    eval_info = eval_policy(args, policy, env, args.seed, '', dump_traj_dir=logger.get_dir())

def run_task(arg_vv, log_dir, exp_name):  # Chester launch

    args = get_args(cmd=False)
    args.plan_horizon=5
    args.max_iters=5
    args.population_size=50
    args.num_elites=10
    
    args.__dict__.update(**arg_vv)

    set_random_seed(args.seed)

    # # Configure logger
    logger.configure(dir=log_dir, exp_name=exp_name)
    log_dir = logger.get_dir()
    assert log_dir is not None
    os.makedirs(log_dir, exist_ok=True)

    wandb.init(project=args.exp_prefix)
    wandb.config.update(args)
    wandb.run.name = "test" if args.run_name == '' else args.run_name
    wandb.run.save()

    # # Dump parameters
    with open(os.path.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2, sort_keys=True)

    # env = make_mp_envs(args.env_name, args.num_env, args.seed)
    env = make(args.env_name)
    set_render_mode(env, args.env_name, 'mesh')
    solve_cem(env, args)

# if __name__ == '__main__':
#     run_task({}, 'data/debug/cem/', 'test')

    
    # if not args.replay:
    #     policy = CEMPolicy(env,
    #                        args.use_mpc,
    #                        plan_horizon=20,
    #                        max_iters=5,
    #                        population_size=50,
    #                        num_elites=5)
    #     # Run policy
    #     env.reset()
    #     initial_state = env.get_state()
    #     action_traj = []
    #     for _ in range(env._max_episode_steps):
    #         action = policy.get_action(obs)
    #         action_traj.append(copy.copy(action))
    #         obs, reward, _, _ = env.step(action)
    #         print('reward:', reward)

    #     traj_dict = {
    #         'initial_state': initial_state,
    #         'action_traj': action_traj
    #     }

    #     with open(traj_path, 'wb') as f:
    #         pickle.dump(traj_dict, f)
    # else:
    #     with open(traj_path, 'rb') as f:
    #         traj_dict = pickle.load(f)
    #     initial_state, action_traj = traj_dict['initial_state'], traj_dict['action_traj']
    #     env.start_record(video_path='./data/videos/', video_name='cem_folding.gif')
    #     env.reset()
    #     env.set_state(initial_state)
    #     for action in action_traj:
    #         env.step(action)
    #     env.end_record()
    # Save the trajectories and replay
