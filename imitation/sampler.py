from imp import init_builtin
import time

import numpy as np
from numpy.core.fromnumeric import shape, size
import torch
from plb.algorithms.bc.bc_agent import Agent as BCAgent
from plb.algorithms.bc.ibc_agent import Agent as IBCAgent
from imitation.utils import get_camera_matrix, get_camera_params, get_partial_pcl, get_partial_pcl2, get_roller_action_from_transform, img_to_tensor, rigid_transform_3D, rotate_y_axis_angle, to_action_mask, LIGHT_DOUGH, LIGHT_TOOL, DARK_DOUGH, DARK_TOOL
from plb.envs.mp_wrapper import SubprocVecEnv
from scipy.spatial.transform import Rotation as R
import os
from chester import logger
device = 'cuda'

def occlude_dough(dough_points, xyz_min, xyz_max):
    ret = dough_points[np.logical_or( np.all((dough_points<xyz_min),axis=1), np.all((dough_points>xyz_max),axis=1))]
    return ret

def sample_traj(env, agent, reset_key, tid, buffer=None, action_mask=None, action_sequence=None, log_succ_score=False, reset_primitive=False, num_moves=1, init='zero', img_size=128):
    """Compute ious: pairwise iou between each pair of timesteps. """
    # assert agent.args.num_env == 1
    states, obses, actions, rewards, succs, scores =[], [], [], [], [0.], [0.]  # Append 0 for the first frame for succs and scores
    reset_obses, reset_infos = [], []
    if action_sequence is None and action_mask is None:
        if tid == 0:
            action_mask = to_action_mask(env, [1, 0])
        else:
            action_mask = to_action_mask(env, [0, 1])
    if isinstance(env, SubprocVecEnv):
        if reset_key is not None:
            state = env.reset([reset_key])[0]
        obs = env.render([{'mode':'rgb','img_size':img_size}])[0]  # rgbd observation
        # T = env.getattr('_max_episode_steps', 0)
    else:
        if reset_key is not None:
            state = env.reset(**reset_key)
        obs = env.render(mode='rgb', img_size=img_size)  # rgbd observation
        # T = env._max_episode_steps
    T = agent.args.buffer_horizon
    if reset_key is not None:
        states.append(state)
        obses.append(obs)
    
    total_r = 0

    total_time = 0
    agent_time = 0
    env_time = 0
    st_time = time.time()
    if not agent.args.open_loop and (isinstance(agent, BCAgent) or isinstance(agent, IBCAgent)): # learner

        view, proj = get_camera_matrix(env)
        if isinstance(env, SubprocVecEnv):
            action_dim = env.getattr("taichi_env.primitives.action_dim", 0)
            _, _, _, mp_info = env.step([np.zeros(action_dim)])
            if reset_primitive:
                primitive_state = env.getfunc('get_primitive_state', 0)
            if reset_key is not None:
                infos = [mp_info[0]]
            else:
                infos = []
        else:
            action_dim = env.taichi_env.primitives.action_dim
            _, _, _, info = env.step(np.zeros(action_dim))
            if reset_primitive:
                primitive_state = env.get_primitive_state
            if reset_key is not None:
                infos = [info]
            else:
                infos = []
        
        frame_stack = agent.args.frame_stack
        # for image agent
        if agent.actor_type == 'Img':
            stack_obs = img_to_tensor(np.array(obs)[None], mode=agent.args.img_mode).to(agent.device)  # stack_obs shape: [1, 4, 64, 64]
            target_img = img_to_tensor(np.array(env.getattr('target_img', 0))[None], mode=agent.args.img_mode).to(agent.device)
            C = stack_obs.shape[1]
            stack_obs = stack_obs.repeat([1, frame_stack, 1, 1])
        # for point cloud agent
        elif agent.args.use_pcl == 'full_pcl':
            target_goal_grid = torch.FloatTensor(buffer.np_target_mass_grids[reset_key['target_v']].reshape(1, -1, 3)).to(agent.device)
            target_goal_grid = torch.cat([target_goal_grid for _ in range(agent.args.frame_stack)], dim=-1)
            stack_states = torch.FloatTensor(np.concatenate([state[:3000].reshape(1, -1, 3), \
                                                        state[3000:3300].reshape(1, -1, 3), ], axis=1)).to(agent.device)
            stack_states = stack_states.repeat([1, 1, frame_stack])
        else:
            goal_pcl = np.zeros((1, 1000, 3))
            if isinstance(env, SubprocVecEnv):
                pcl = get_partial_pcl2(np.array(env.getattr('target_img', 0)), LIGHT_DOUGH, DARK_DOUGH, view, proj)[:, :3]
            else:
                pcl = get_partial_pcl2(np.array(env.target_img), LIGHT_DOUGH, DARK_DOUGH, view, proj)[:, :3]
            if len(pcl) > 1000:
                rand = np.random.choice(len(pcl), size=1000, replace=False)
                pcl = pcl[rand]
            goal_pcl[0, :len(pcl)] = pcl
            goal_pcl = torch.FloatTensor(goal_pcl).to(agent.device)
            goal_pcl_len = torch.tensor([[len(pcl)]]).to(agent.device)
        with torch.no_grad():
            filenames = []
            for i in range(T):
                t1 = time.time()

                with torch.no_grad():
                    if agent.args.env_name in ['RollTest-v1', 'RollLong-v1', 'RollTest-v2', 'RollExp-v1', 'RollDev-v1'] and (i>=50 and i<120):
                        if i < 100:
                            action = (-1*actions[50-(i-49)])
                            if i >= 80:
                                action[2] = 0.
                        else:
                            action = np.array([0, 1, 0, 0, 0, 0], dtype=float)
                    elif agent.args.env_name == 'RollTest-v3' and (i >= 50 and i < 120):
                        if i < 100:
                            action = (-1*actions[50-(i-49)])
                            if i >= 80:
                                action[2] = 0.
                        else:
                            action = np.array([0, 0, 0, 0, -0.05, 0], dtype=float)
                    else:
                        if agent.actor_type == 'Img':
                            action, done = agent.act(stack_obs, target_img, tid)
                            obs_tensor = img_to_tensor(np.array(obs)[None], mode=agent.args.img_mode).to(agent.device)
                            stack_obs = torch.cat([stack_obs, obs_tensor], dim=1)[:, -frame_stack * C:]
                        elif agent.args.use_pcl =='full_pcl':
                            action, done = agent.act(stack_states, target_goal_grid, tid, stats=buffer.stats)
                            state_tensor = torch.FloatTensor(np.concatenate([state[:3000].reshape(1, -1, 3), \
                                                            state[3000:3300].reshape(1, -1, 3), ], axis=1)).to(agent.device)
                            stack_states = torch.cat([stack_states, state_tensor], dim=-1)[:, :, -frame_stack*3:]
                            if False:
                                import matplotlib.pyplot as plt
                                fig = plt.figure(figsize=(8,8))
                                ax = fig.add_subplot(111, projection='3d')
                                ax.view_init(140, -90)
                                ax.set_xlim3d(0.2, 0.8)
                                ax.set_ylim3d(0, 0.5)
                                ax.set_zlim3d(0.2, 0.8)
                                ax.scatter(stack_states[0,:1000,0].cpu(), stack_states[0,:1000,1].cpu(), stack_states[0,:1000,2].cpu(), marker='o', alpha=0.5, s=2)
                                ax.scatter(stack_states[0,1000:1100,0].cpu(), stack_states[0,1000:1100,1].cpu(), stack_states[0,1000:1100,2].cpu(), marker='o', alpha=0.5, s=2)
                                ax.scatter(target_goal_grid[0,:,0].cpu(), target_goal_grid[0,:,1].cpu(), target_goal_grid[0,:,2].cpu(), marker='x', alpha=0.5, s=2)
                                ax.set_xlabel('x')
                                ax.set_ylabel('y')
                                ax.set_zlabel('z')
                                # create file name and append it to a list
                                filename = f'{i}.png'

                                filenames.append(filename)
                                plt.show()
                                breakpoint()
                        else:
                            # dough
                            dough_pcl = np.zeros((1,1000,3))
                            pcl1 = get_partial_pcl2(obs, LIGHT_DOUGH, DARK_DOUGH, view, proj, random_mask=False, p=0.)[:,:3]
                            if len(pcl1) > 1000:
                                rand = np.random.choice(len(pcl1), size=1000, replace=False)
                                pcl1 = pcl1[rand]
                            tool_pcl = np.zeros((1,100,3))
                            # tool
                            if agent.args.gt_tool:
                                pcl2 = state[3000:3300].reshape(-1, 3)
                            else:
                                pcl2 = get_partial_pcl2(obs, LIGHT_TOOL, DARK_TOOL, view, proj)[:,:3]
                            if len(pcl2) > 100:
                                rand = np.random.choice(len(pcl2), size=100, replace=False)
                                pcl2 = pcl2[rand]
                            dough_pcl[0, :len(pcl1)] = pcl1
                            tool_pcl[0, :len(pcl2)] = pcl2
                            dough_pcl = torch.FloatTensor(dough_pcl).to(agent.device)
                            tool_pcl = torch.FloatTensor(tool_pcl).to(agent.device)
                            obs_pcl = torch.cat([dough_pcl, tool_pcl], dim=1)
                            obs_pcl_len = torch.tensor([[len(pcl1), len(pcl2)]]).to(agent.device)
                            # frame
                            if agent.args.frame == 'tool':
                                tool_xyz = state[3300:3303].reshape(1, 3)
                                pcl1 -= tool_xyz
                                pcl2 -= tool_xyz
                                pcl -= tool_xyz
                                tool_xyz = torch.FloatTensor(tool_xyz).to(agent.device)
                                action, done = agent.act_partial_pcl(obs_pcl, obs_pcl_len, goal_pcl, goal_pcl_len, tid, eval=agent.args.rm_eval_noise, tool_xyz=tool_xyz)
                            elif agent.args.actor_type == 'PointActorToolParticle':
                                pred_tool_flow = agent.act_partial_pcl(obs_pcl, obs_pcl_len, goal_pcl, goal_pcl_len, tid, eval=agent.args.rm_eval_noise)
                                pred_tool_flow = pred_tool_flow[0].detach().cpu().numpy()
                                # visualize it 
                                # rigid_transform_3D
                                if False:
                                    import matplotlib.pyplot as plt
                                    fig = plt.figure(figsize=(8,8))
                                    ax = fig.add_subplot(111, projection='3d')
                                    ax.view_init(140, -90)
                                    ax.set_xlim3d(0.2, 0.8)
                                    ax.set_ylim3d(0, 0.5)
                                    ax.set_zlim3d(0.2, 0.8)
                                    ax.scatter(pcl1[:,0], pcl1[:,1], pcl1[:,2], marker='o', alpha=0.5, s=2)
                                    ax.scatter(pcl2[:,0], pcl2[:,1], pcl2[:,2], marker='o', alpha=0.5, s=2)
                                    ax.scatter(pcl[:,0], pcl[:,1], pcl[:,2], marker='x', alpha=0.5, s=2)
                                    ax.set_xlabel('x')
                                    ax.set_ylabel('y')
                                    ax.set_zlabel('z')
                                    ax.quiver(pcl2[:,0], pcl2[:,1], pcl2[:,2], pred_tool_flow[:,0], pred_tool_flow[:,1], pred_tool_flow[:,2], 
                                    length=0.1, normalize=True, color='red')
                                    plt.show()
                                    # print("predicted flow:", pred_tool_flow[:10])

                                r, t = rigid_transform_3D(pcl2.T, (pcl2+pred_tool_flow).T)
                                action = torch.from_numpy(get_roller_action_from_transform(r, t).reshape(1, 6))
                                # print("predicted action:", action)
                                done = torch.zeros(size=(1,1))
                            else:
                                action, done = agent.act_partial_pcl(obs_pcl, obs_pcl_len, goal_pcl, goal_pcl_len, tid, eval=agent.args.rm_eval_noise)
                            if False:
                                import matplotlib.pyplot as plt
                                fig = plt.figure(figsize=(8,8))
                                ax = fig.add_subplot(111, projection='3d')
                                ax.view_init(140, -90)
                                ax.set_xlim3d(0.2, 0.8)
                                ax.set_ylim3d(0, 0.5)
                                ax.set_zlim3d(0.2, 0.8)
                                ax.scatter(pcl1[:,0], pcl1[:,1], pcl1[:,2], marker='o', alpha=0.5, s=2)
                                ax.scatter(pcl2[:,0], pcl2[:,1], pcl2[:,2], marker='o', alpha=0.5, s=2)
                                ax.scatter(pcl[:,0], pcl[:,1], pcl[:,2], marker='x', alpha=0.5, s=2)
                                ax.set_xlabel('x')
                                ax.set_ylabel('y')
                                ax.set_zlabel('z')
                                # create file name and append it to a list
                                filename = f'{i}.png'

                                filenames.append(filename)
                                plt.show()
                                # breakpoint()
                            #     plt.savefig(os.path.join('data/debug/pcl', filename))
                            #     plt.close()
                            # breakpoint()
                        action = action[0].detach().cpu().numpy()
                        done = done[0].detach().cpu().numpy()
                if np.round(done).astype(int) == 1 and agent.terminate_early:
                    break
                t2 = time.time()
                if isinstance(env, SubprocVecEnv):
                    mp_next_state, mp_reward, _, mp_info = env.step([action])
                    state, reward, info = mp_next_state[0], mp_reward[0], mp_info[0]
                    obs = env.render([{'mode':'rgb','img_size':img_size}])[0]
                else:
                    state, reward, _, info = env.step(action)
                    obs = env.render(mode='rgb',img_size=img_size)
                t3 = time.time()

                agent_time += t2 - t1
                env_time += t3 - t2

                actions.append(action)
                states.append(state)
                obses.append(obs)
                infos.append(info)
                total_r += reward
                rewards.append(reward)
                # if log_succ_score:
                #     succs.append(succ)
                #     scores.append(score)
            if reset_primitive:
                if isinstance(env, SubprocVecEnv):
                    _, reset_obses, _, _ = env.getfunc('primitive_reset_to', 0, list_kwargs=[{'idx': tid, 'reset_states': primitive_state}])  # TODO tid
                else:
                    _, reset_obses, _, _ = env.primitive_reset_to(tid, primitive_state)
                for obs in reset_obses:
                    obs_tensor = img_to_tensor(np.array(obs)[None], mode=agent.args.img_mode).to(agent.device)
                    # stack_obs = torch.cat([stack_obs, obs_tensor], dim=1)[:, -frame_stack * C:]
                    stack_obs = obs_tensor
                    assert frame_stack == 1
                    if log_succ_score:
                        with torch.no_grad():
                            z_obs, _, _ = agent.vae.encode(stack_obs)
                            z_goal, _, _ = agent.vae.encode(target_img)
                            if i == 0:
                                z_init = z_obs.clone()
                            succ = agent.fea_pred(z_init, z_obs, tid, type='succ', eval=True).detach().cpu().numpy()[0]
                            score = agent.fea_pred(z_obs, z_goal, tid, type='score', eval=True).detach().cpu().numpy()[0]
                    obses.append(obs)
                    if log_succ_score:
                        succs.append(succ)
                        scores.append(score)
        if isinstance(env, SubprocVecEnv):
            target_img = np.array(env.getattr('target_img', 0))
        else:
            target_img = np.array(env.target_img)

    elif action_sequence is not None or agent.args.open_loop:
        view, proj = get_camera_matrix(env)
        if isinstance(env, SubprocVecEnv):
            action_dim = env.getattr("taichi_env.primitives.action_dim", 0)
            _, _, _, mp_info = env.step([np.zeros(action_dim)])
            if reset_primitive:
                primitive_state = env.getfunc('get_primitive_state', 0)
            if reset_key is not None:
                infos = [mp_info[0]]
            else:
                infos = []
        else:
            action_dim = env.taichi_env.primitives.action_dim
            _, _, _, info = env.step(np.zeros(action_dim))
            if reset_primitive:
                primitive_state = env.get_primitive_state
            if reset_key is not None:
                infos = [info]
            else:
                infos = []
        if agent.args.open_loop:
            goal_pcl = np.zeros((1, 1000, 3))
            if isinstance(env, SubprocVecEnv):
                pcl = get_partial_pcl2(np.array(env.getattr('target_img', 0)), LIGHT_DOUGH, DARK_DOUGH, view, proj)[:, :3]
            else:
                pcl = get_partial_pcl2(np.array(env.target_img), LIGHT_DOUGH, DARK_DOUGH, view, proj)[:, :3]
            if len(pcl) > 1000:
                rand = np.random.choice(len(pcl), size=1000, replace=False)
                pcl = pcl[rand]
            goal_pcl[0, :len(pcl)] = pcl
            goal_pcl = torch.FloatTensor(goal_pcl).to(agent.device)
            goal_pcl_len = torch.tensor([[len(pcl)]]).to(agent.device)
            dough_pcl = np.zeros((1,1000,3))
            pcl1 = get_partial_pcl2(obs, LIGHT_DOUGH, DARK_DOUGH, view, proj, random_mask=False, p=0.)[:,:3]
            if len(pcl1) > 1000:
                rand = np.random.choice(len(pcl1), size=1000, replace=False)
                pcl1 = pcl1[rand]
            tool_pcl = np.zeros((1,100,3))
            # tool
            if agent.args.gt_tool:
                pcl2 = state[3000:3300].reshape(-1, 3)
            else:
                pcl2 = get_partial_pcl2(obs, LIGHT_TOOL, DARK_TOOL, view, proj)[:,:3]
            if len(pcl2) > 100:
                rand = np.random.choice(len(pcl2), size=100, replace=False)
                pcl2 = pcl2[rand]
            dough_pcl[0, :len(pcl1)] = pcl1
            tool_pcl[0, :len(pcl2)] = pcl2
            dough_pcl = torch.FloatTensor(dough_pcl).to(agent.device)
            tool_pcl = torch.FloatTensor(tool_pcl).to(agent.device)
            obs_pcl = torch.cat([dough_pcl, tool_pcl], dim=1)
            obs_pcl_len = torch.tensor([[len(pcl1), len(pcl2)]]).to(agent.device)
            action_sequence, done = agent.act_partial_pcl(obs_pcl, obs_pcl_len, goal_pcl, goal_pcl_len, tid, eval=agent.args.rm_eval_noise)
            action_sequence = action_sequence[0].detach().cpu().numpy().reshape(-1, 6)
        i = 0
        for t in range(T):
            t1 = time.time()

            with torch.no_grad():
                if agent.args.env_name in ['RollTest-v1', 'RollLong-v1', 'RollTest-v2', 'RollExp-v1'] and (t>=50 and t<120):
                    if t < 100:
                        action = (-1*action_sequence[50-(t-49)])
                        if t >= 80:
                            action[2] = 0.
                    else:
                        action = np.array([0, 1, 0, 0, 0, 0], dtype=float)
                else:
                    action = action_sequence[i]
                    i += 1
                t2 = time.time()
                if isinstance(env, SubprocVecEnv):
                    mp_next_state, mp_reward, _, mp_info = env.step([action])
                    state, reward, info = mp_next_state[0], mp_reward[0], mp_info[0]
                    obs = env.render([{'mode':'rgb','img_size':img_size}])[0]
                else:
                    state, reward, _, info = env.step(action)
                    obs = env.render(mode='rgb',img_size=img_size)
                t3 = time.time()

                agent_time += t2 - t1
                env_time += t3 - t2

                actions.append(action)
                states.append(state)
                obses.append(obs)
                infos.append(info)
                total_r += reward
                rewards.append(reward)
                # mass_grids.append(info['mass_grid'])
        if isinstance(env, SubprocVecEnv):
            target_img = np.array(env.getattr('target_img', 0))
        else:
            target_img = np.array(env.target_img)
    else:
        # Solver
        taichi_env = env.taichi_env
        action_dim = taichi_env.primitives.action_dim
        _, _, _, info = env.step(np.zeros(action_dim))

        primitive_states = env.get_primitive_state()
        print(primitive_states)
        infos = [info]
        actions = []
        for move in range(num_moves):
            load = np.load('init_actions.npy')
            init_action = np.zeros([T, taichi_env.primitives.action_dim],dtype=np.float32)
            init_action[:50] = load[:50]
            init_action[50:100] = load[-50:]
            # for t in agent.reset_ts:
            #     init_action[t-20:t] = load[-20:]
            # if init == 'zero':
            #     init_action = np.zeros([T, taichi_env.primitives.action_dim],dtype=np.float32)
            # elif init == 'normal':
            #     init_action = np.random.normal(size=(T, taichi_env.primitives.action_dim))
            # if buffer != None:
            #     init_action[:50] = buffer.buffer['actions'][:50]
                # init_action[:20] = np.tile(np.array([0., 0., -0.5, 0., 0., 0.]), (20, 1))
                # init_action[100:120] = np.tile(np.array([0., 1., -0.5, 0., 0., 0.]), (20, 1))
                # breakpoint()
            info = agent.solve(init_action, action_mask=action_mask, loss_fn=taichi_env.compute_loss, max_iter=agent.args.gd_max_iter, lr=agent.args.lr)
            agent.save_plot_buffer(logger.get_dir())
            if agent.args.debug_gradient:
                agent.plot_grad(os.path.join(logger.get_dir(), 'gradient_curve.png'))
            actions.extend(info['best_action'])
            agent_time = time.time() - st_time
            ## Execution
            for i in range(T):
                t1 = time.time()
                if agent.args.opt_mode == 'multireset' and i in agent.reset_ts:
                    reset_idx = agent.reset_ts.index(i)
                    new_reset_states = rotate_y_axis_angle(primitive_states, 0, agent.reset_angs[reset_idx])
                    new_reset_states[0][1] -= 0.1
                    next_state, reward, _, info = env.env.step(actions[i], state=new_reset_states)
                else:
                    next_state, reward, _, info = env.step(actions[i])

                infos.append(info)
                env_time += time.time() - t1
                states.append(next_state)
                obs = taichi_env.render(mode='rgb', img_size=img_size)
                obses.append(obs)
                total_r += reward
                rewards.append(reward)

                # if i == 100:
                #     print("MPC!")
                #     init_action = np.zeros([T-100, taichi_env.primitives.action_dim],dtype=np.float32)
                #     info = agent.solve(init_action, action_mask=action_mask, loss_fn=taichi_env.compute_loss, max_iter=agent.args.gd_max_iter2, lr=agent.args.lr)
                #     agent.save_plot_buffer(logger.get_dir(), postfix='2')
                #     actions.extend(info['best_action'])
                #     agent_time = time.time() - st_time

            target_img = env.target_img
            if reset_primitive and move < num_moves - 1:
                # state_copy = next_state[-16:-9]
                # p = R.from_quat([state_copy[4], state_copy[5], state_copy[6], state_copy[3]])
                # q = R.from_rotvec(np.pi/2 * np.array([0, 1, 0])) # hard coded rotate 90 around y-axis for now
                # quat = R.as_quat(q*p)
                # reset_state = np.array([primitive_states[0][0], primitive_states[0][1], primitive_states[0][2], quat[-1], quat[0], quat[1], quat[2]])
                # reset_states, _, reset_obs, _, reset_info = env.primitive_reset_to(idx=tid, reset_states=primitive_state)
                # states[-1] = reset_states[-1]
                # obses[-1] = reset_obs[-1]
                # infos[-1] = reset_info[-1]
                # reset_obses.append(reset_obs)
                # reset_infos.append(reset_info)
                env.taichi_env.primitives[0].set_state(0, primitive_states[0])
    # import imageio
    # with imageio.get_writer(os.path.join('data/debug/pcl','partial_pcl.gif'), mode='I') as writer:
    #     for filename in filenames:
    #         image = imageio.imread(os.path.join('data/debug/pcl',filename))
    #         writer.append_data(image)
    
    emds = np.array([info['info_emd'] for info in infos])
    if len(infos) > 0:
        info_normalized_performance = np.array([info['info_normalized_performance'] for info in infos])
        info_final_normalized_performance = info_normalized_performance[-1]
    else:
        info_normalized_performance = []
        info_final_normalized_performance = None

    total_time = time.time() - st_time
    ret = {'states': np.array(states).astype(np.float32),
           'obses': np.array(obses).astype(np.float32),
           'actions': np.array(actions).astype(np.float32),
           'target_img': target_img,
           'rewards': np.array(rewards),
           'info_rewards': np.array(rewards),
           'info_emds': emds,
           'info_final_normalized_performance': info_final_normalized_performance,
           'info_normalized_performance': info_normalized_performance,
           'info_total_r': total_r,
           'info_total_time': total_time,
           'info_agent_time': agent_time,
           'info_env_time': env_time,
           'action_mask': action_mask}
    if log_succ_score:
        ret['succs'] = np.array(succs)  # Should miss the first frame
        ret['scores'] = np.array(scores)
    if reset_key is not None:
        ret.update(**reset_key)
    if False:
        ### for resets
        max_reset_length = 20
        obs_shape, dtype = np.array(obses[0]).shape, np.array(obses[0]).dtype
        reset_motion_obses = np.zeros([num_moves, max_reset_length, *obs_shape], dtype=dtype)
        reset_info_emds = np.zeros([num_moves, max_reset_length], dtype=np.float32)
        reset_motion_lens = np.zeros(shape=(num_moves,), dtype=np.int32)
        for i in range(num_moves):
            reset_ob = reset_obses[i]
            if len(reset_ob) == 0:
                continue
            reset_emds = np.array([info['info_emd'] for info in reset_infos[i]])
            if len(reset_ob) > max_reset_length:
                idx = sorted(np.random.choice(range(len(reset_ob)), max_reset_length, replace=False))
                reset_ob = reset_ob[idx]
                reset_emds = reset_emds[idx]
            reset_motion_obses[i, :len(reset_ob)] = reset_ob
            reset_info_emds[i, :len(reset_ob)] = reset_emds
            reset_motion_lens[i] = len(reset_ob)
        ret.update({'reset_motion_obses':reset_motion_obses, 
                    'reset_motion_lens':reset_motion_lens,
                    'reset_info_emds':reset_info_emds,
        })
    return ret
