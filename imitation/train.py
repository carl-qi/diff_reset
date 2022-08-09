import taichi
from imitation.agent import Agent
from imitation.sampler import sample_traj
from imitation.buffer import ReplayBuffer
from imitation.utils import aggregate_traj_info
from plb.engine.taichi_env import TaichiEnv
from plb.optimizer.solver import Solver

import argparse
import random
import numpy as np
import torch
import json
import os
from chester import logger

from plb.envs import make
from plb.algorithms.logger import Logger
from plb.utils.visualization_utils import save_numpy_as_gif, save_numpy_as_img, cv_render
from imitation.utils import to_action_mask, visualize_dataset, get_iou, load_target_info
from imitation.utils import img_to_tensor, img_to_np, visualize_trajs, batch_rand_int


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_args(cmd=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='Roll-v1')
    parser.add_argument('--cached_state_path', type=str, default='./datasets/1006_Roll')
    parser.add_argument('--algo', type=str, default='imitation')
    parser.add_argument('--dataset_name', type=str, default='tmp')
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--sdf_loss", type=float, default=10)
    parser.add_argument("--density_loss", type=float, default=10)
    parser.add_argument("--contact_loss", type=float, default=1)
    parser.add_argument("--chamfer_loss", type=float, default=0.)
    parser.add_argument("--soft_contact_loss", action='store_true')
    parser.add_argument("--gd_num_steps", type=int, default=50, help="steps for the gradient descent(gd) expert")

    # differentiable physics parameters
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--softness", type=float, default=666.)
    parser.add_argument("--optim", type=str, default='Adam', choices=['Adam', 'Momentum'])

    parser.add_argument("--num_trajs", type=int, default=20)

    # Actor
    parser.add_argument("--feature_dim", type=int, default=50)
    parser.add_argument("--il_num_epoch", type=int, default=1000)
    parser.add_argument("--il_lr", type=float, default=1e-3)
    parser.add_argument("--il_eval_freq", type=int, default=10)
    parser.add_argument("--image_dim", type=int, default=64)
    parser.add_argument("--img_mode", type=str, default='rgbd')
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--frame_stack", type=int, default=1)

    # feasibility
    parser.add_argument("--pos_ratio", type=float, default=0.5)
    parser.add_argument("--bin_succ", type=bool, default=False)

    # encoder (VAE)
    parser.add_argument("--encoder_lr", type=float, default=1e-3)

    parser.add_argument("--debug_overfit_test", type=bool, default=False)
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
        traj = sample_traj(env, agent, reset_key, tid=tid, compute_ious=False, compute_target_ious=True,
                           target_img=target_imgs[reset_key['target_v']], target_mass_grids=np_target_mass_grids)
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

    device = 'cuda'

    # Configure logger
    logger.configure(dir=log_dir, exp_name=exp_name)
    log_dir = logger.get_dir()
    assert log_dir is not None
    os.makedirs(log_dir, exist_ok=True)

    # Dump parameters
    with open(os.path.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2, sort_keys=True)

    # ----------preparation done------------------

    buffer = ReplayBuffer(arg_vv)
    def get_state_goal_id(traj_id):
        goal_id = traj_id % 200  # There are 200 targets and # TODO hardcoded goal and state number
        state_id = (traj_id * 19941019) % 300
        return {'init_v': state_id, 'target_v': goal_id}  # state and target version

    if 'agent_path' in args.__dict__:
        agent_vv_path = os.path.join(os.path.dirname(args.agent_path), 'variant.json')
        print(agent_vv_path)
        with open(agent_vv_path, 'r') as f:
            agent_vv = json.load(f)
        args.img_mode = agent_vv['img_mode']  # Overrid image mode
        args.frame_stack = agent_vv['frame_stack']
        print('Using image mode from the loaded agent: ', agent_vv['img_mode'])

    target_info = load_target_info(args, device)
    np_target_imgs, target_imgs, np_target_mass_grids = target_info['np_target_imgs'], target_info['target_imgs'], target_info['np_target_mass_grids']
    if args.task in ['gen_data', 'train_policy', 'train_feas']:
        obs_channel = len(args.img_mode) * args.frame_stack
        img_obs_shape = (args.image_dim, args.image_dim, obs_channel)

        if args.gd_num_steps is None:
            args.gd_num_steps = 50 * 200

        if args.chamfer_loss > 0.:
            args.density_loss = args.sdf_loss = 0.

        env = make(args.env_name, nn=(args.algo == 'nn'))
        env.seed(args.seed)
        taichi_env: TaichiEnv = env.unwrapped.taichi_env
        T = args.buffer_horizon
        action_dim = taichi_env.primitives.action_dim

        solver = Solver(args, taichi_env, (0,), return_dist=True)
        agent = Agent(args, solver, img_obs_shape, action_dim, num_tools=2, device=device)

        if args.task != 'gen_data':
            env.taichi_env.loss.set_target_update(False)  # Save speed on reset

        if args.task == 'gen_data':
            args.dataset_path = os.path.join(logger.get_dir(), 'dataset.gz')
            traj_ids = np.array_split(np.arange(args.num_trajs), args.gen_num_batch)[args.gen_batch_id]

            for traj_id in traj_ids:
                reset_key = get_state_goal_id(traj_id)

                traj = sample_traj(env, solver, reset_key, action_mask=to_action_mask([0, 1]), target_mass_grids=np_target_mass_grids)
                print(
                    f"traj {traj_id}, agent time: {traj['info_agent_time']}, env time: {traj['info_env_time']}, total time: {traj['info_total_time']}")
                buffer.add(traj)

            for traj_id in traj_ids:
                reset_key = get_state_goal_id(traj_id)
                traj = sample_traj(env, solver, reset_key, action_mask=to_action_mask([1, 0]), target_mass_grids=np_target_mass_grids)
                print(
                    f"traj {traj_id}, agent time: {traj['info_agent_time']}, env time: {traj['info_env_time']}, total time: {traj['info_total_time']}")
                buffer.add(traj)

                buffer.save(os.path.join(args.dataset_path))
            visualize_dataset(args.dataset_path, args.cached_state_path, os.path.join(logger.get_dir(), 'visualization.gif'),
                              overlay_target=True)
            exit()
        elif args.task == 'train_policy':
            buffer.load(args.dataset_path)
            all_action_losses, all_done_losses = [], []
            tool_idxes = buffer.get_tool_idxes()

            N = min(len(tool_idxes[i]) for i in range(len(tool_idxes)))  # Should be the same
            num_batch = N // args.batch_size

            if args.debug_overfit_test:  # Set the idx to be the ones sampled from the specified trajectories
                horizon = 50
                traj_ids = args.debug_overfit_traj_ids
                debug_idxes = []
                for traj_id in traj_ids:
                    debug_idxes.append(np.arange(traj_id * horizon, traj_id * horizon + horizon))
                debug_idxes = np.array(debug_idxes).flatten()

            for epoch in range(args.il_num_epoch):
                print('epoch: ', epoch)

                batch_idxes = (np.random.permutation(N)[:num_batch * args.batch_size]).reshape([num_batch, args.batch_size])

                def sample_goal_obs(buffer, obs_idx, hindsight_goal_ratio):
                    n = len(obs_idx)
                    horizon = 50

                    traj_id = obs_idx // horizon
                    traj_t = obs_idx % horizon
                    # reset_key = get_state_goal_id(traj_id)
                    # init_v, target_v = reset_key['init_v'], reset_key['target_v']
                    init_v, target_v = buffer.buffer['init_v'][obs_idx, 0], buffer.buffer['target_v'][obs_idx, 0]
                    hindsight_future_idx = batch_rand_int(traj_t, horizon, n)
                    optimal_step = buffer.get_optimal_step()
                    hindsight_flag = np.round(np.random.random(n) < hindsight_goal_ratio).astype(np.int32)
                    hindsight_goal_imgs = buffer.buffer['obses'][hindsight_future_idx + traj_id * horizon]
                    target_goal_imgs = np_target_imgs[target_v]
                    goal_obs = img_to_tensor(hindsight_flag[:, None, None, None] * hindsight_goal_imgs +
                                             (1 - hindsight_flag[:, None, None, None]) * target_goal_imgs, mode=args.img_mode).to(device)
                    hindsight_done_flag = (hindsight_future_idx == horizon - 1).astype(np.int32)
                    target_done_flag = (traj_t == optimal_step[traj_id, traj_t]).astype(np.int32)
                    done_flag = hindsight_flag * hindsight_done_flag + (1 - hindsight_flag) * target_done_flag
                    done_flag = torch.FloatTensor(done_flag).to(device)
                    return goal_obs, done_flag

                for curr_idx in batch_idxes:
                    t_obses, t_goal_obses, t_actions, t_dones = [], [], [], []
                    for tid in [0, 1]:  # For each tool
                        curr_tool_idx = tool_idxes[tid][curr_idx]
                        if args.debug_overfit_test:
                            curr_tool_idx = debug_idxes
                        obs = img_to_tensor(buffer.sample_stacked_obs(curr_tool_idx, args.frame_stack), mode=args.img_mode).to(device)
                        action = torch.FloatTensor(buffer.buffer['actions'][curr_tool_idx]).to(device)
                        goal_obs, done = sample_goal_obs(buffer, curr_tool_idx, args.hindsight_goal_ratio)
                        t_obses.append(obs.clone())
                        t_goal_obses.append(goal_obs.clone())
                        t_actions.append(action)
                        t_dones.append(done)
                    train_info_actor = agent.train_actor(t_obses, t_goal_obses, t_dones, expert_actions=t_actions)
                    all_action_losses.append(train_info_actor['avg_action_loss'])
                    all_done_losses.append(train_info_actor['avg_done_loss'])

                if epoch % args.il_eval_freq == 0:
                    if args.debug_overfit_test:
                        trajs, eval_info = eval_training_traj(args.debug_overfit_traj_ids, args, buffer, env, agent, target_imgs, np_target_imgs,
                                                              np_target_mass_grids, save_name='eval_{}.gif'.format(epoch))
                        trajs[0]['train_action'] = buffer.buffer['actions'][curr_tool_idx]
                        trajs[0]['train_obs'] = buffer.sample_stacked_obs(curr_tool_idx, args.frame_stack)
                        import pickle
                        with open(os.path.join(logger.get_dir(), f'eval_{epoch}.pkl'), 'wb') as f:
                            pickle.dump(trajs, f)

                    else:
                        horizon = 50
                        num_traj = len(buffer) // horizon
                        eval_trajs = np.random.randint(0, num_traj, size=8)
                        trajs, eval_info = eval_training_traj(eval_trajs, args, buffer, env, agent, target_imgs, np_target_imgs,
                                                              np_target_mass_grids, save_name='eval_{}.gif'.format(epoch))

                    infos = aggregate_traj_info(trajs)
                    infos.update(**eval_info)
                    logger.record_tabular("epoch", epoch)
                    for key, val in infos.items():
                        logger.record_tabular(key, val)
                    logger.record_tabular('avg_action_loss', np.array(all_action_losses).mean())
                    logger.record_tabular('avg_done_loss', np.array(all_done_losses).mean())
                    logger.dump_tabular()
                    all_action_losses, all_done_losses = [], []
                    agent.save(os.path.join(logger.get_dir(), f'agent_{epoch}.ckpt'))
        elif args.task == 'train_feas':
            horizon = 50
            agent.load_actor(args.agent_path)

            if args.gen_agent_dataset:
                print('start generating trajectory')
                agent.terminate_early = False  # Collecting trajectories with a fixed length of {horizon}
                # Generate the agent dataset in batch
                traj_ids = np.array_split(np.arange(args.num_trajs), args.gen_num_batch)[args.gen_batch_id]

                args.agent_dataset_path = os.path.join(logger.get_dir(), 'agent_dataset.gz')
                # Collect dataset

                for traj_id in traj_ids:
                    print('traj id:', traj_id)
                    reset_key = get_state_goal_id(traj_id)

                    traj = sample_traj(env, agent, reset_key, tid=0, action_mask=to_action_mask([0, 1]), compute_ious=True, compute_target_ious=True,
                                       target_img=target_imgs[reset_key['target_v']], target_mass_grids=np_target_mass_grids)
                    buffer.add(traj)
                buffer.save(os.path.join(args.agent_dataset_path), save_mass_grid=False)

                for traj_id in traj_ids:
                    print('traj id:', traj_id)
                    reset_key = get_state_goal_id(traj_id)
                    traj = sample_traj(env, agent, reset_key, tid=1, action_mask=to_action_mask([1, 0]), compute_ious=True, compute_target_ious=True,
                                       target_img=target_imgs[reset_key['target_v']], target_mass_grids=np_target_mass_grids)
                    buffer.add(traj)

                buffer.save(os.path.join(args.agent_dataset_path), save_mass_grid=False)
                visualize_dataset(args.agent_dataset_path, args.cached_state_path, os.path.join(logger.get_dir(), 'visualization.gif'),
                                  overlay_target=True)
                exit()
            else:
                buffer.load(args.agent_dataset_path, filename='agent_dataset.gz')
                buffer.generate_train_eval_split()

            optimal_step = buffer.get_optimal_step().flatten()

            all_train_info = []
            tool_idxes = buffer.get_tool_idxes(buffer.train_idx)

            # Due to computation, only select 8 of the valid trajecotires
            num_eval_traj = 8
            eval_traj_idx = np.random.choice(buffer.eval_traj_idx, num_eval_traj, replace=False)
            train_traj_idx = np.random.choice(buffer.train_traj_idx, num_eval_traj, replace=False)

            N = min(len(tool_idxes[i]) for i in range(len(tool_idxes)))  # May not be the same.
            num_batch = N // args.batch_size

            for epoch in range(args.il_num_epoch):
                print('epoch: ', epoch)
                batch_idxes = (np.random.permutation(N)[:num_batch * args.batch_size]).reshape([num_batch, args.batch_size])

                for curr_idx in batch_idxes:
                    t_obses, t_succ_obses, t_score_obses, t_succ_labels, t_score_labels = [], [], [], [], []
                    for tid in [0, 1]:  # For each tool
                        curr_tool_idx = tool_idxes[tid][curr_idx]
                        obs = img_to_tensor(buffer.buffer['obses'][curr_tool_idx], mode=args.img_mode).to(device)
                        init_v, target_v = buffer.buffer['init_v'][curr_tool_idx, 0], buffer.buffer['target_v'][curr_tool_idx, 0]
                        goal_obs = target_imgs[target_v]

                        # Generating success
                        # positive pairs (o_i, o_j) from the same trajectory, negative pairs (o_i, o_j) from different trajectories
                        def sample_positive_idx(buffer, obs_idx):
                            n = len(obs_idx)
                            horizon = 50
                            traj_id = obs_idx // horizon
                            traj_t = obs_idx % horizon
                            future_idx = batch_rand_int(traj_t, horizon, n) + traj_id * horizon
                            return future_idx

                        def sample_negative_idx(buffer, obs_idx, all_traj_idx):
                            horizon = 50
                            n = len(obs_idx)
                            filter_traj_id = obs_idx // horizon  # traj_id to be filtered
                            all_traj_id = np.array([x for x in all_traj_idx if x not in filter_traj_id])
                            traj_id = all_traj_id[np.random.randint(0, len(all_traj_id), n)]
                            traj_t = np.random.randint(0, horizon, n)
                            neg_idx = traj_id * horizon + traj_t
                            return neg_idx

                        num_pos = int(args.pos_ratio * len(curr_tool_idx))
                        num_neg = len(curr_tool_idx) - num_pos
                        neg_idx = sample_negative_idx(buffer, curr_tool_idx[:num_neg], buffer.train_traj_idx)
                        pos_idx = sample_positive_idx(buffer, curr_tool_idx[num_neg:])
                        succ_obs = img_to_tensor(buffer.buffer['obses'][np.concatenate([neg_idx, pos_idx], axis=0)], args.img_mode).to(device,
                                                                                                                                       non_blocking=True)
                        succ_label = torch.cat([torch.zeros(size=(num_neg,), device=device, dtype=torch.int),
                                                torch.ones(size=(num_pos,), device=device, dtype=torch.int)])
                        # Generating scores
                        # Pairs sampled from o_t, g
                        score_obs = goal_obs
                        score_label = buffer.buffer['target_ious'][curr_tool_idx]
                        score_label = torch.FloatTensor(score_label).to(device)

                        t_obses.append(obs)
                        t_succ_obses.append(succ_obs)
                        t_score_obses.append(score_obs)
                        t_succ_labels.append(succ_label)
                        t_score_labels.append(score_label)

                    train_info_actor = agent.train_feas(t_obses, t_succ_obses, t_score_obses, t_succ_labels, t_score_labels)
                    all_train_info.append(train_info_actor)

                if epoch % args.il_eval_freq == 0:
                    from plb.utils.visualization_utils import make_grid, save_rgb
                    from imitation.utils import write_number

                    for name, traj_idx in zip(['train', 'eval'], [train_traj_idx, eval_traj_idx]):
                        test_init_v = buffer.buffer['init_v'][traj_idx * horizon, 0]
                        test_target_v = buffer.buffer['target_v'][traj_idx * horizon, 0]
                        action_masks = buffer.buffer['action_mask'][traj_idx * horizon]
                        tids = buffer.get_tid(action_masks)
                        pred_ious, gt_ious, obs_imgs, goal_imgs = [], [], [], []
                        all_idxes = []
                        for i, (traj_id, init_v, target_v, tid) in enumerate(zip(traj_idx, test_init_v, test_target_v, tids)):
                            buffer_ious = buffer.buffer['target_ious'][traj_id * horizon:traj_id * horizon + horizon].flatten()
                            obs = buffer.buffer['obses'][traj_id * horizon: traj_id * horizon + horizon]
                            obs_imgs.append(obs.copy())
                            all_idxes.append(np.arange(traj_id * horizon, traj_id * horizon + horizon))
                            tensor_obs = img_to_tensor(np.array(obs), args.img_mode).to(device)
                            with torch.no_grad():
                                ious = agent.fea_pred(tensor_obs, target_imgs[target_v][None].repeat(len(tensor_obs), 1, 1, 1), tid,
                                                      type='score', eval=True).detach().cpu().numpy()
                            pred_ious.append(ious)
                            gt_ious.append(buffer_ious)
                            goal_imgs.append(np.tile(np_target_imgs[target_v][None, :, :, :], [horizon, 1, 1, 1]).copy())
                        pred_ious = np.array(pred_ious)  # N x 50
                        obs_imgs = np.array(obs_imgs)  # N x 50
                        goal_imgs = np.array(goal_imgs)
                        gt_ious = np.array(gt_ious)
                        all_idxes = np.array(all_idxes).flatten()
                        save_imgs = []
                        for i in range(len(traj_idx)):
                            for t in range(horizon):
                                overlayed_img = obs_imgs[i, t] * 0.7 + goal_imgs[i, t] * 0.3
                                i1, i2 = overlayed_img.copy(), overlayed_img.copy()
                                write_number(i1, pred_ious[i, t])
                                write_number(i2, gt_ious[i, t])
                                save_imgs.append(np.hstack([i1, i2]))
                        img_shape = save_imgs[0].shape
                        save_imgs = np.array(save_imgs).reshape(len(traj_idx), horizon, *img_shape)
                        imgs = []
                        for t in range(horizon):
                            grid_img = make_grid(np.array(save_imgs[:, t]), ncol=4, padding=3, pad_value=0.5)
                            imgs.append(grid_img)
                        save_numpy_as_gif(np.array(imgs) * 255., os.path.join(log_dir, f"{name}_pred_score_{epoch}.gif"))

                        # Evaluating success

                        n = len(all_idxes)
                        num_eval = 20
                        obs_idx = all_idxes[np.random.randint(0, n, num_eval)]
                        pos_idx = sample_positive_idx(buffer, obs_idx)
                        neg_idx = sample_negative_idx(buffer, obs_idx, buffer.eval_traj_idx)

                        tensor_obses = img_to_tensor(np.vstack([buffer.buffer['obses'][obs_idx], buffer.buffer['obses'][obs_idx]]), args.img_mode).to(
                            device, non_blocking=True)
                        tensor_goals = img_to_tensor(np.vstack([buffer.buffer['obses'][pos_idx], buffer.buffer['obses'][neg_idx]]), args.img_mode).to(
                            device, non_blocking=True)
                        gt_succ = np.vstack([np.ones(len(pos_idx)), np.zeros(len(neg_idx))]).flatten()
                        all_succ = []
                        for tid in [0, 1]:
                            succ = agent.fea_pred(tensor_obses, tensor_goals, tid, type='succ', eval=True).detach().cpu().numpy()
                            all_succ.append(succ)

                        np_obses, np_goals = img_to_np(tensor_obses), img_to_np(tensor_goals)
                        imgs = []
                        pred_succ = []
                        for i, (obs, goal, gt_s) in enumerate(zip(np_obses, np_goals, gt_succ)):
                            i1, i2 = obs.copy(), goal.copy()
                            write_number(i1, "{}_{}".format(int(round(all_succ[0][i])), int(round(all_succ[1][i]))))
                            write_number(i2, int(gt_s))
                            imgs.append(np.hstack([i1, i2]))
                            pred_succ.append(max(all_succ[0][i], all_succ[1][i]))  # Reachibility as the max of the two tools

                        grid_img = make_grid(np.array(imgs), ncol=10, padding=3, pad_value=0.5)
                        save_rgb(os.path.join(log_dir, f"{name}_pred_succ_{epoch}.png"), grid_img)

                        score_mse = np.mean(np.square(pred_ious.flatten() - gt_ious.flatten()))
                        succ_accuracy = 1 - np.mean(np.abs(np.array(pred_succ) - gt_succ))
                        logger.record_tabular(f'{name}_score_mse', score_mse)
                        logger.record_tabular(f'{name}_succ_accuracy', succ_accuracy)

                    # from imitation.test_trajectories import test_init_v, test_target_v
                    # imgs, feas = [], []
                    # for tid in [0, 1]:
                    #     for (init_v, target_v) in zip(test_init_v, test_target_v):
                    #         env.reset(init_v=init_v, target_v=target_v)
                    #
                    #         render_kwargs = {'mode': 'rgb', 'target': False}
                    #         obs = taichi_env.render(**render_kwargs)
                    #         tensor_obs = img_to_tensor(np.array(obs[None]), args.img_mode).to(device)
                    #         with torch.no_grad():
                    #             iou = agent.fea_pred(tensor_obs, target_imgs[target_v][None], tid)[0].item()
                    #         img = obs * 0.7 + np_target_imgs[target_v] * 0.3
                    #         imgs.append(img.copy())
                    #         feas.append(iou)
                    #
                    # for img, fea in zip(imgs, feas):
                    #     write_number(img, fea)
                    # grid_img = make_grid(np.array(imgs), ncol=len(imgs) // 2, padding=3, pad_value=0.5)
                    # save_rgb(os.path.join(log_dir, 'test_{}.png'.format(epoch)), grid_img)

                    logger.record_tabular("epoch", epoch)
                    agg_info = aggregate_traj_info(all_train_info)
                    for key in agg_info.keys():
                        if 'mean' in key:
                            logger.record_tabular(key, agg_info[key])
                    logger.dump_tabular()
                    agent.save(os.path.join(logger.get_dir(), f'agent_{epoch}.ckpt'))
        else:
            raise NotImplementedError

    elif args.task == 'train_encoder':
        from imitation.encoder.vae import VAE
        from plb.utils.visualization_utils import make_grid

        dataset_folder = os.path.dirname(args.dataset_path)
        print('Training encoder using dataset ' + dataset_folder)

        buffer.load(os.path.join(args.dataset_path))
        buffer.generate_train_eval_split()
        train_idx, eval_idx = buffer.train_idx, buffer.eval_idx
        obs_channel = len(args.img_mode)

        encoder = VAE(image_channels=obs_channel).to(device)
        optim = torch.optim.Adam(lr=args.encoder_lr, params=encoder.parameters())
        criterion = torch.nn.MSELoss()

        num_batch = len(train_idx) // args.batch_size

        eval_vis_idx = np.random.randint(0, len(eval_idx), 12)
        target_vis_idx = np.random.randint(0, len(target_imgs), 12)
        for epoch in range(args.num_epoch):
            batch_idxes = (train_idx[np.random.permutation(len(train_idx))][:num_batch * args.batch_size]).reshape([num_batch, args.batch_size])
            train_losses, train_bce_losses, train_kl_losses = [], [], []
            for batch_idx in batch_idxes:
                obses = img_to_tensor(buffer.buffer['obses'][batch_idx], mode=args.img_mode).to(device)
                reconstr_obses, mu, logvar = encoder.reconstr(obses)
                loss, bce_loss, kl_loss = encoder.loss_fn(reconstr_obses, obses, mu, logvar, beta=args.encoder_beta)
                optim.zero_grad()
                loss.backward()
                optim.step()
                train_losses.append(loss.item())
                train_bce_losses.append(bce_loss.item())
                train_kl_losses.append(kl_loss.item())

            if epoch % 5 == 0:
                torch.save(encoder.state_dict(), os.path.join(logger.get_dir(), f'encoder_{epoch}.pth'))

            # Evaluate
            with torch.no_grad():
                M = len(eval_idx)
                eval_losses = []
                gt_imgs, pred_imgs = [], []
                for i in range(0, M, args.batch_size):
                    obses = img_to_tensor(buffer.buffer['obses'][eval_idx[i:min(i + args.batch_size, M)]], mode=args.img_mode).to(device)
                    reconstr_obses, _, _ = encoder.reconstr(obses)
                    eval_losses.append(criterion(obses, reconstr_obses).item())
                    gt_imgs.append(img_to_np(obses))
                    pred_imgs.append(img_to_np(reconstr_obses))
                gt_imgs, pred_imgs = np.concatenate(gt_imgs, axis=0), np.concatenate(pred_imgs, axis=0)

                img_array = np.concatenate([gt_imgs[eval_vis_idx], pred_imgs[eval_vis_idx]], axis=0)
                img_array = make_grid(img_array, ncol=12, padding=3, pad_value=0.5)
                save_numpy_as_img(img_array, os.path.join(logger.get_dir(), f'eval_{epoch}.png'))

            with torch.no_grad():
                obses = target_imgs[target_vis_idx]
                reconstr_obses, _, _ = encoder.reconstr(obses)
                img_array = np.concatenate([img_to_np(obses), img_to_np(reconstr_obses)], axis=0)

                img_array = make_grid(img_array, ncol=12, padding=3, pad_value=0.5)
                save_numpy_as_img(img_array, os.path.join(logger.get_dir(), f'target_{epoch}.png'))

            logger.record_tabular('epoch', epoch)
            logger.record_tabular('train_loss', np.array(train_losses).mean())
            logger.record_tabular('train_bce_loss', np.array(train_bce_losses).mean())
            logger.record_tabular('train_kl_loss', np.array(train_kl_losses).mean())
            logger.record_tabular('eval_loss', np.array(eval_losses).mean())
            logger.dump_tabular()


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
            'dataset_path': '/home/jianrenw/carl/research/dev/PlasticineLab/data/local/1020_Roll_exp_working/1020_Roll_exp_working_2021_10_21_11_26_07_0001/dataset.gz'}
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
