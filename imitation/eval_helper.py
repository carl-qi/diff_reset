from imitation.sampler import sample_traj
from imitation.utils import visualize_trajs, aggregate_traj_info, make_grid, img_to_tensor, img_to_np
from plb.utils.visualization_utils import save_rgb
from imitation.hardcoded_eval_trajs import get_eval_traj
from chester import logger
import matplotlib.pyplot as plt
import os.path as osp
import numpy as np
from tqdm import tqdm
import os
from imitation.env_spec import get_threshold

def dict_add_prefix(d, pre):
    new_d = {}
    for key, val in d.items():
        new_d[pre + key] = val
    return new_d


def eval_skills(args, env, agent, epoch, tids=None):
    """ Run each skill on evaluation configurations; Save videos;
    Return raw trajs indexed by tid, time_step; Return aggregated info"""
    init_vs, target_vs = get_eval_traj(args.cached_state_path)
    skill_traj, skill_info = [], {}
    tids = list(range(2)) if tids is None else tids
    for tid in tids:
        trajs = []
        for init_v, target_v in tqdm(zip(init_vs, target_vs), desc="eval skills"):
            reset_key = {'init_v': init_v, 'target_v': target_v}
            traj = sample_traj(env, agent, reset_key, tid, log_succ_score=True, reset_primitive=True)
            trajs.append(traj)
        keys = ['info_normalized_performance', 'succs', 'scores', 'score_error']
        fig, axes = plt.subplots(1, len(keys), figsize=(len(keys)* 5, 5))
        for key_id, key in enumerate(keys):
            # Visualize traj
            if key == 'info_normalized_performance':
                visualize_trajs(trajs, key=key, ncol=10, save_name=osp.join(logger.get_dir(), f"eval_skill_inp_epoch_{epoch}_{tid}.gif"),
                                vis_target=True)
            elif key !='score_error':
                visualize_trajs(trajs, key=key, ncol=10, save_name=osp.join(logger.get_dir(), f"eval_skill_{key}_epoch_{epoch}_{tid}.gif"),
                                vis_target=True)
            # Visualize stats
            for traj_id, traj in enumerate(trajs):
                if key =='score_error':
                    vals = np.abs(traj['info_emds'] + traj['scores'][:len(traj['info_emds'])])
                else:
                    vals = traj[key]
                axes[key_id].plot(range(len(vals)), vals, label=f'traj_{traj_id}')
            axes[key_id].set_title(key)
        plt.legend()
        plt.tight_layout()
        plt.savefig(osp.join(logger.get_dir(), f"eval_skill_stats_epoch_{epoch}_{tid}.png"))


        info = aggregate_traj_info(trajs)
        skill_info.update(**dict_add_prefix(info, f'eval/skill_{tid}/'))
        skill_traj.append(trajs)

    return skill_traj, skill_info


def eval_vae(args, agent, skill_traj, epoch):
    all_obses = []
    for tid in range(args.num_tools):
        for i in range(len(skill_traj[tid])):
            for j in range(len(skill_traj[tid][i]['obses'])):
                all_obses.append(skill_traj[tid][i]['obses'][j])
    all_obses = np.array(all_obses)
    N = len(all_obses)
    sample_idx = np.random.randint(0, N, 8)

    obses = img_to_tensor(all_obses[sample_idx], mode=args.img_mode).to(agent.device)
    reconstr_obses, _, _ = agent.vae.reconstr(obses)
    reconstr_obses = img_to_np(reconstr_obses)
    imgs = np.concatenate([all_obses[sample_idx], reconstr_obses], axis=2)
    mse = np.mean(np.square(all_obses[sample_idx] - reconstr_obses))

    img = make_grid(imgs, ncol=4, padding=3)
    save_rgb(osp.join(logger.get_dir(), f'vae_reconstr_{epoch}.png'), img)
    return {'eval/vae_reconstr_error': mse}


def eval_plan(args, env, agent, epoch, demo=False):
    demo = True # For quick ICLR experiment
    from imitation.compose_skills import plan, visualize_all_traj, visualize_adam_info, visualize_mgoal, execute

    init_vs, target_vs = get_eval_traj(args.cached_state_path)

    save_dir = os.path.join(logger.get_dir(), f'plan_epoch_{epoch}')
    os.makedirs(save_dir, exist_ok=True)

    normalized_scores = []
    for i, (init_v, target_v) in enumerate(tqdm(zip(init_vs, target_vs), desc="plan")):
        plan_info = {'env': env, 'init_v': init_v, 'target_v': target_v}
        best_traj, all_traj, traj_info = plan(args, agent, plan_info, plan_step=args.plan_step, opt_mode='adam')

        if not demo:
            img, sorted_idxes = visualize_all_traj(all_traj, overlay=False)
            save_grid_img = make_grid(img[:10], ncol=5, padding=5, pad_value=0.5)
            save_rgb(osp.join(save_dir, f'plan_traj_{i}.png'), save_grid_img[:, :, :3])
        else:
            img, sorted_idxes = visualize_all_traj(all_traj, overlay=False, demo=True)
            for ii , iimg in enumerate(img[:10]):
                save_rgb(osp.join(save_dir, f'plan_traj_{i}_sol_{ii}.png'), np.array(iimg[:, :, :3]).astype(np.float32))

        execute_name = osp.join(save_dir, f'execute_{i}.gif')
        _, score= execute(env, agent, best_traj, save_name=execute_name, reset_primitive=True, demo=demo)
        normalized_scores.append(score)
        visualize_adam_info(traj_info, savename=osp.join(save_dir, f'adam_{i}.png'), topk=8)
        if 'goal_his' in traj_info:
            goal_his = np.concatenate(traj_info['goal_his'], axis=1)  # traj_info goal his: [4 (list), 200, 10, img_dims]
            goal_his = goal_his[:, sorted_idxes[:10]]
            visualize_mgoal(goal_his, savename=osp.join(save_dir, f'adam_{i}.gif'))
    normalized_score = np.array(normalized_scores)
    thr = get_threshold(args.env_name)
    success = np.array(normalized_score > thr).astype(float)
    return {'plan_normalized_score': np.mean(normalized_score),
            'plan_success': np.mean(success),
            'all_normalized_score': normalized_scores,
            'all_success': success}
