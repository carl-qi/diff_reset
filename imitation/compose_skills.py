import os.path
import pickle
import copy
import numpy as np
import torch
from imitation.utils import img_to_tensor, write_number
from plb.utils.visualization_utils import save_numpy_as_gif, make_grid
from imitation.utils import get_iou, img_to_np, traj_gaussian_logl, gaussian_logl
import os.path as osp
from tqdm import tqdm


def plan(args, agent, plan_info, plan_step=2, opt_mode='search'):
    """plan info should contain either (1) obs and goal_obs  (2) env, init_v and target_v, 'np_target_imgs"""
    assert opt_mode in ['search', 'adam', 'sample']
    if 'obs' in plan_info and 'goal_obs' in plan_info:
        obs, goal_obs = plan_info['obs'], plan_info['goal_obs']
        assert isinstance(obs, np.ndarray)
    else:
        env = plan_info['env']
        env.reset([{'init_v': plan_info['init_v'], 'target_v': plan_info['target_v']}])
        obs = env.render(mode='rgb')[0]
        goal_obs = env.getattr('target_img', 0)

    tensor_obs = img_to_tensor(np.array(obs[None]), agent.args.img_mode).to(agent.device, non_blocking=True)
    tensor_goal_obs = img_to_tensor(np.array(goal_obs[None]), agent.args.img_mode).to(agent.device, non_blocking=True)

    best_traj = None
    all_traj = []

    if opt_mode == 'search':
        mid_goals = plan_info['mid_goals']  # All intermediate goals to search over
        search_dims = [len(mid_goals)] * plan_step  # N-1 intermediate goals + planned reached goal
        search_idxes = np.indices(search_dims).reshape(plan_step, -1).transpose()
        tensor_mid_goals = img_to_tensor(mid_goals, agent.args.img_mode).to(agent.device)

    info = {}
    search_tids_idxes = np.indices([args.num_tools] * plan_step).reshape(plan_step, -1).transpose()
    for tids in search_tids_idxes:
        if opt_mode == 'search':
            all_succ = []
            for step in range(plan_step):
                if step == 0:
                    curr_obs = tensor_obs.repeat([search_idxes.shape[0], 1, 1, 1])
                else:
                    curr_obs = tensor_mid_goals[search_idxes[:, step - 1]]
                curr_goal = tensor_mid_goals[search_idxes[:, step]]
                with torch.no_grad():
                    succ = agent.fea_pred(curr_obs, curr_goal, tids[step], type='succ', eval=True)
                    all_succ.append(succ[None, :])

            with torch.no_grad():
                score = agent.fea_pred(curr_goal, tensor_goal_obs.repeat([search_idxes.shape[0], 1, 1, 1]), tids[-1], type='score',
                                       eval=True)  # Either tid is fine. Actually we don't need separate scoring networks
            all_succ = torch.cat(all_succ, dim=0)  # [plan_step, num_plans]
            f = all_succ[0]
            for i in range(1, len(all_succ)):
                f = f * all_succ[i]
            f = f * score
            np_f, np_score, np_all_succ = f.detach().cpu().numpy(), score.detach().cpu().numpy(), all_succ.detach().cpu().numpy()
            for i in range(len(search_idxes)):
                traj_imgs = [obs] + [mid_goals[search_idxes[i][j]] for j in range(plan_step)]
                traj_imgs.append(goal_obs)
                traj = {
                    'score': np_f[i],
                    'traj_succ': np_all_succ[:, i],
                    'traj_score': score[i],
                    'tool': tids,
                    'traj_img': traj_imgs,
                    'init_v': plan_info['init_v'],
                    'target_v': plan_info['target_v']
                }
                if best_traj is None or traj['score'] > best_traj['score']:
                    best_traj = traj
                all_traj.append(traj)
        elif opt_mode == 'adam':
            # Preparation
            n = args.adam_sample
            niter = args.adam_iter
            zdim = agent.vae.z_dim
            z = agent.vae.sample_latents(n * plan_step, agent.device).reshape(n, plan_step, zdim)
            z.requires_grad = True
            optim = torch.optim.Adam(params=[z], lr=args.adam_lr)

            # Get latent code of initial and goal observation
            with torch.no_grad():
                z_obs, _, _ = agent.vae.encode(tensor_obs)
                z_goal_obs, _, _ = agent.vae.encode(tensor_goal_obs)
                tiled_z_obs, tiled_z_goal_obs = z_obs.repeat([n, 1]), z_goal_obs.repeat([n, 1])

            traj_succ_his, traj_score_his, score_his, loss_his, goal_his, zlogl_his = [], [], [], [], [], []
            for i in tqdm(range(niter), desc='Adam during planning'):
                # Don't need to decode anymore
                # mgoals = agent.vae.decode(z.view(n * plan_step, zdim))
                # mgoals = mgoals.view(n, plan_step, *mgoals.shape[1:])

                if args.save_goal_his:
                    raise NotImplementedError
                    # np_mgoals = img_to_np(mgoals.view(n * plan_step, *mgoals.shape[2:])).reshape(n, plan_step, *obs.shape)

                all_succ = []
                for step, tid in enumerate(tids):
                    if step == 0:
                        curr_z = tiled_z_obs
                    else:
                        curr_z = z[:, step - 1]
                    curr_goal = z[:, step]
                    succ = agent.fea_pred(curr_z, curr_goal, tids[step], type='succ', eval=True)
                    all_succ.append(succ[None, :])
                all_succ = torch.cat(all_succ, dim=0)  # [plan_step, num_plans]
                score = agent.fea_pred(z[:, len(tids) - 1], tiled_z_goal_obs, tids[-1], type='score', eval=True)
                score = torch.min(score, torch.zeros_like(score))  # score is negative emd, which should be less than zero
                f = all_succ[0]
                for i in range(1, len(all_succ)):
                    f = f * torch.max(all_succ[i], torch.ones_like(all_succ[i]) * 5e-2)
                f = f * torch.exp(score * 10.)
                traj_zlogl = traj_gaussian_logl(z)
                # loss = -torch.sum(f)
                # loss = -torch.sum(f) - 0.01 * torch.sum(all_succ[0]) - 0.01 * torch.sum(all_succ[1])
                loss = -torch.sum(f) - 0.01 * torch.sum(all_succ[0]) - 0.01 * torch.sum(all_succ[1])
                # loss = -torch.sum(f) - 0.01 * torch.sum(all_succ[0]) - 0.01 * torch.sum(all_succ[1]) - 0.0005 * torch.sum(traj_zlogl)
                # loss = -torch.sum(f) - 0.001 * torch.sum(zlogl)
                # loss = -torch.sum(score)
                optim.zero_grad()
                loss.backward()
                optim.step()

                # Projection to the constraint set
                if 'min_zlogl' is not None:
                    with torch.no_grad():
                        z_logl = gaussian_logl(z)
                        projected_z = z / torch.max(torch.sqrt((z_logl[:, :, None] / args.min_zlogl)), torch.ones(1, device=z.device))
                        z.copy_(projected_z)

                traj_succ_his.append(all_succ.detach().cpu().numpy())
                traj_score_his.append(score.detach().cpu().numpy())
                score_his.append(f.detach().cpu().numpy())
                loss_his.append(loss.item())
                zlogl_his.append(traj_zlogl.detach().cpu().numpy())
                if args.save_goal_his:
                    raise NotImplementedError
                    goal_his.append(np_mgoals)
            score = score_his[-1]
            idxes = np.argsort(score)[::-1][:10]  # Get the top 10 result

            np_mgoals = img_to_np(agent.vae.decode(z.view(n * plan_step, zdim))).reshape([n, plan_step, *obs.shape])

            for i in idxes:
                traj_img = [obs] + list(np_mgoals[i, :]) + [goal_obs]
                traj = {
                    'score': score[i],
                    'traj_succ': traj_succ_his[-1][:, i],
                    'traj_score': traj_score_his[-1][i],
                    'tool': tids,
                    'traj_img': traj_img,
                    'init_v': plan_info['init_v'],
                    'target_v': plan_info['target_v']
                }
                if best_traj is None or traj['score'] > best_traj['score']:
                    best_traj = traj
                all_traj.append(traj)
            if 'loss_his' in info.keys():
                info['loss_his'].append(np.array(loss_his))
                info['score_his'].append(np.array(score_his))  # num_iter x num_traj
                if args.save_goal_his:
                    info['goal_his'].append(np.array(goal_his))
                info['his_name'].append(f'{tids[0]}_{tids[1]}')
                info['zlogl_his'].append(np.array(zlogl_his))
            else:
                info['loss_his'] = [np.array(loss_his)]
                info['score_his'] = [np.array(score_his)]
                if args.save_goal_his:
                    info['goal_his'] = [np.array(goal_his)]
                info['his_name'] = [f'{tids[0]}_{tids[1]}']
                info['zlogl_his'] = [np.array(zlogl_his)]

    return best_traj, all_traj, info


def execute(env, agent, plan_traj, reset_primitive=False, save_name=None, demo=False):
    env.reset([{'init_v': plan_traj['init_v'], 'target_v': plan_traj['target_v']}])
    device, img_mode = agent.device, agent.args.img_mode
    imgs, all_actions = [], []
    for step in range(1, len(plan_traj['traj_img']) - 1):  # Skip the first observation since we only need the goals
        np_goal = plan_traj['traj_img'][step]
        tensor_goal = img_to_tensor(np_goal[None], img_mode).to(device)
        tid = plan_traj['tool'][step - 1]
        print('tool id:', tid)
        primitive_state = env.getfunc('get_primitive_state', 0)
        for i in range(50):
            obs = env.render(mode='rgb')[0]
            tensor_obs = img_to_tensor(np.array(obs[None]), img_mode).to(device)
            action, done = agent.act(tensor_obs, tensor_goal, tid)
            action, done = action[0].detach().cpu().numpy(), done[0].detach().cpu().numpy()
            if np.round(done).astype(int) == 1 and agent.terminate_early and i > 0:
                break
            _, _, _, infos = env.step([action])
            info = infos[0]
            if not demo:
                curr_img = (obs * 0.7 + np_goal * 0.3).copy()
            else:
                curr_img = obs
            write_number(curr_img, info['info_normalized_performance'], color=(1,1,1))
            imgs.append(curr_img)
            all_actions.append(action)
        # For ablation
        if agent.args.num_tools == 1:
            actions, obses, _, _ = env.getfunc('primitive_reset_to', 0, list_kwargs=[{'idx': 0, 'reset_states': primitive_state}])
            actions, obses, _, _ = env.getfunc('primitive_reset_to', 0, list_kwargs=[{'idx': 1, 'reset_states': primitive_state}])
        else:
            actions, obses, _, _ = env.getfunc('primitive_reset_to', 0, list_kwargs=[{'idx': tid, 'reset_states': primitive_state}])
        for obs, action in zip(obses, actions):
            if demo:
                write_number(obs, info['info_normalized_performance'], color = (1,1,1))
            imgs.append(obs)
            all_actions.append(action)
    for i in range(20):
        imgs.append(curr_img)
        all_actions.append(np.zeros_like(action))

    if save_name is not None:
        save_numpy_as_gif(np.array(imgs) * 255, save_name)
        pkl_name = copy.copy(save_name)[:-4] + '.pkl'
        plan_traj['actions'] = np.array(all_actions)
        with open(pkl_name, 'wb') as f:
            pickle.dump(plan_traj, f)
    return imgs, info['info_normalized_performance']


def visualize_all_traj(all_traj, overlay=False, demo=False):
    all_traj_imgs = []
    all_score = []
    for traj in all_traj:
        imgs = traj['traj_img']
        traj_imgs = []
        for i in range(len(imgs) - 1):
            if overlay:
                overlay_img = (imgs[i] * 0.7 + imgs[i + 1] * 0.3).copy()
            else:
                overlay_img = imgs[i].copy()
            if not demo:
                if i == len(imgs) - 2:
                    write_number(overlay_img, traj['traj_score'])
                else:
                    write_number(overlay_img, traj['traj_succ'][i])
            traj_imgs.append(overlay_img)
        last_goal = imgs[-1].copy()
        if not demo:
            write_number(last_goal, traj['score'])
        traj_imgs.append(last_goal)

        traj_imgs = np.hstack(traj_imgs)
        all_traj_imgs.append(traj_imgs)
        all_score.append(traj['score'])
    idx = np.argsort(np.array(all_score))[::-1]  # Sort by the scores
    all_traj_imgs = np.array(all_traj_imgs)[idx]
    return all_traj_imgs, idx


import matplotlib.pyplot as plt


def visualize_adam_info(traj_info, savename, topk=None):
    """If topk is not None, only plot the loss and zlogl for the topk trajectories"""
    tool_idx = 0  # TODO change this later too.
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    num_iter = len(traj_info['loss_his'][tool_idx])
    num_traj = len(traj_info['score_his'][tool_idx][0])  # score_his: num_iter x num_traj
    ax1.plot(range(num_iter), np.array(traj_info['loss_his'][tool_idx]))
    ax1.set_title('total_loss')
    if topk is None:
        for i in range(num_traj):
            ax2.plot(range(num_iter), traj_info['score_his'][tool_idx][:, i], label=f'traj_{i}')
            ax3.plot(range(num_iter), traj_info['zlogl_his'][tool_idx][:, i], label=f'traj_{i}')
    else:
        final_score = traj_info['score_his'][tool_idx][-1, :]
        idxes = np.argsort(final_score)[::-1][:topk]
        for i in range(topk):
            ax2.plot(range(num_iter), traj_info['score_his'][tool_idx][:, idxes[i]], label=f'traj_{idxes[i]}')
            ax3.plot(range(num_iter), traj_info['zlogl_his'][tool_idx][:, idxes[i]], label=f'traj_{idxes[i]}')
    ax2.set_title('scores')
    ax3.set_title('z_logl')
    plt.legend()
    plt.tight_layout()
    plt.savefig(savename)


def visualize_mgoal(goal_his, savename):
    # goal his shape: list of size niter,  num_sample x num_step x 64 x 64 x 4
    imgs = []
    for i in range(0, goal_his.shape[0], 5):
        mgoal = goal_his[i].transpose(1, 0, 2, 3, 4)
        mgoal = np.concatenate(list(mgoal), axis=2)
        grid_img = make_grid(mgoal, ncol=5, padding=5, pad_value=0.5)
        write_number(grid_img, i)
        imgs.append(grid_img)
    for j in range(20):
        imgs.append(grid_img)

    save_numpy_as_gif(np.array(imgs) * 255, savename)
