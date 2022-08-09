from imitation.agent import Agent
from imitation.sampler import sample_traj
from imitation.imitation_buffer import ImitationReplayBuffer
from imitation.utils import load_target_info, visualize_trajs
from imitation.train_full import get_args, set_random_seed
from imitation.eval_helper import get_eval_traj
from plb.envs.mp_wrapper import make_mp_envs
from imitation.env_spec import get_threshold
from tqdm import tqdm
import json
import os
import os.path as osp
from chester import logger
import numpy as np


def eval_skill_direct_execution(args, env, agent):
    """ Run each skill on evaluation configurations; Save videos;
    Return raw trajs indexed by tid, time_step; Return aggregated info"""
    init_vs, target_vs = get_eval_traj(args.cached_state_path)

    search_tids_idxes = np.indices([args.num_tools] * args.plan_step).reshape(args.plan_step, -1).transpose()
    key = 'info_normalized_performance'
    combo_performances, combo_names = [], []
    for tids in search_tids_idxes:
        combo_name = ""
        for tid in tids:
            combo_name += str(tid)
        all_trajs = []
        for init_v, target_v in tqdm(zip(init_vs, target_vs), desc="eval skills"):
            reset_key = {'init_v': init_v, 'target_v': target_v}

            _ = env.reset([reset_key])[0]
            action_dim = env.getattr("taichi_env.primitives.action_dim", 0)
            _, _, _, mp_info = env.step([np.zeros(action_dim)])
            info = mp_info[0]

            obses, vals = [np.array(env.render(mode='rgb'))], [[info[key]]]
            for tid in tids:
                skill_traj = sample_traj(env, agent, None, tid, log_succ_score=False, reset_primitive=True)
                if len(skill_traj[key]) < len(skill_traj['obses']):  # Add reset info
                    val = np.concatenate([skill_traj[key], np.tile(skill_traj[key][-1], len(skill_traj['obses']) - len(skill_traj[key]))])
                else:
                    val = skill_traj[key]
                if len(val) > 0:
                    vals.append(val)
                if len(skill_traj['obses']) > 0:
                    obses.append(skill_traj['obses'])
            all_trajs.append(
                {'obses': np.concatenate(obses),
                 'target_img': skill_traj['target_img'],
                 key: np.concatenate(vals)})
        visualize_trajs(all_trajs, key=key, ncol=len(init_vs), save_name=osp.join(logger.get_dir(), f"eval_skill_inp_{combo_name}.gif"),
                        vis_target=True)
        combo_names.append(combo_name)
        combo_performances.append([traj[key][-1] for traj in all_trajs])
    combo_performances = np.array(combo_performances)
    thr = get_threshold(args.env_name)
    combo_success = np.array(combo_performances > thr)
    # combo_performances: (num_combo x num_test_traj, i.e. 4 x 5)
    return {'combo_best_performance_mean': np.mean(np.max(combo_performances, axis=0)),
            'combo_best_performance_std': np.std(np.max(combo_performances, axis=0)),
            'combo_best_success_mean': np.mean(np.max(combo_success, axis=0)),
            'combo_avg_performance_mean': np.mean(combo_performances),
            'combo_avg_performance_std': np.mean(combo_performances),
            'combo_avg_success_mean': np.mean(combo_success)}


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
    # Dump parameters
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
    buffer.generate_train_eval_split()
    target_info = load_target_info(args, device)
    buffer.__dict__.update(**target_info)

    # # ----------preparation done------------------
    agent = Agent(args, None, img_obs_shape, action_dim, num_tools=args.num_tools, device=device)
    if args.resume_path is not None:
        agent.load(args.resume_path)

    info = eval_skill_direct_execution(args, env, agent)
    for key, val in info.items():
        logger.record_tabular(key, val)
    logger.dump_tabular()

    env.close()
