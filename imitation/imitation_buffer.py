import torch
import numpy as np
from imitation.utils import batch_rand_int, get_camera_params, get_partial_pcl, get_partial_pcl2, img_to_tensor, img_to_np, LIGHT_TOOL, LIGHT_DOUGH, DARK_DOUGH, DARK_TOOL
from imitation.buffer import ReplayBuffer


def filter_buffer_nan(buffer):
    actions = buffer.buffer['actions']
    idx = np.where(np.isnan(actions))
    print('{} nan actions detected. making them zero.'.format(len(idx[0])))
    buffer.buffer['actions'][idx] = 0.

def filter_hard_coded_actions(buffer, start, end):
    print("filtering out hard coded actions")
    idx_to_keep = [i for i in range(buffer.maxlen) if i<buffer.cur_size and (i % buffer.horizon < start or i % buffer.horizon >= end)]
    for key in buffer.buffer:
        # hack for now, don't filter the reset images because later we need to plot them
        if key != 'obses' and buffer.buffer[key].shape[0] == buffer.maxlen:
            buffer.buffer[key][:len(idx_to_keep)] = buffer.buffer[key][idx_to_keep]
    buffer.cur_size = len(idx_to_keep)
    buffer.cur_pt = buffer.cur_size
    print("new buffer size: ", buffer.cur_size)

def segment_partial_pcl(buffer, key, light, dark, view, proj):
    max_pt = buffer.max_pts[key]
    if key in buffer.buffer.keys():
        print(key + " is already in buffer, returning!")
        return
    else:
        print("generating partial point cloud for: ", key)
        buffer.buffer[key] = np.zeros(shape=(buffer.cur_size, max_pt, 3), dtype=np.float)
        buffer.buffer[key+'_len'] = np.zeros(shape=(buffer.cur_size, 1), dtype=np.int32)
        for i in range(buffer.cur_size):
            if key == 'goal_pcl':
                target_v = buffer.buffer['target_v'][i]
                pcl = get_partial_pcl2(buffer.np_target_imgs[target_v], light, dark, view, proj)[:, :3]
            else:
                pcl = get_partial_pcl2(buffer.buffer['obses'][i], light, dark, view, proj)[:, :3]
            if len(pcl) > max_pt:
                rand = np.random.choice(len(pcl), size=max_pt, replace=False)
                pcl = pcl[rand]
            buffer.buffer[key][i, :pcl.shape[0]] = pcl
            buffer.buffer[key+'_len'][i] = pcl.shape[0]

class ImitationReplayBuffer(ReplayBuffer):
    def __init__(self, *args, **kwargs):
        super(ImitationReplayBuffer, self).__init__(*args, **kwargs)
        self.tool_idxes = {}
        self.stats = {}
        self.cfg = {}

    def get_tool_idxes(self, tid, idxes=None):
        """ Return categorized time index within the given idxes based on its tool"""
        action_mask = self.buffer['action_mask']
        if idxes is None:
            idxes = np.arange(self.cur_size)
        if tid == 0:
            return idxes[np.where(action_mask[idxes, 0] > 0.5)[0]]
        elif tid == 1:
            return idxes[np.where(action_mask[idxes, 0] < 0.5)[0]]

    # Should also be consistent with the one when generating the feasibility prediction dataset! line 291 of train.py
    def get_tid(self, action_mask):
        """ Should be consistent with get_tool_idxes"""
        if len(action_mask.shape) == 1:
            return int(action_mask[0] < 0.5)
        elif len(action_mask.shape) == 2:
            return np.array(action_mask[:, 0] < 0.5).astype(np.int)

    def get_optimal_step(self):
        """ For each trajectory, compute its optimal time step - return [0, horizon] for each one """
        if not hasattr(self, 'optimal_step'):
            emd_rewards = -self.buffer['info_emds'][:self.cur_size].reshape(-1, self.horizon)
            N = emd_rewards.shape[0]
            # For each step in the buffer, find its optimal future step.
            optimal_step = np.zeros((N, self.horizon), dtype=np.int32)
            for i in range(N):
                max_iou = -1.
                max_step = None
                for j in reversed(range(self.horizon)):
                    if emd_rewards[i, j] >= max_iou or max_step is None:
                        max_step = j
                        max_iou = emd_rewards[i, j]
                    optimal_step[i, j] = max_step
            self.optimal_step = optimal_step
        return self.optimal_step

    def get_epoch_tool_idx(self, epoch, tid):
        # Get a index generator for each tool index of the mini-batch
        # Note: Assume static buffer
        if tid not in self.tool_idxes:
            tool_idxes = self.get_tool_idxes(tid,idxes=self.train_idx)
            # Shuffle the tool idxes so that if the trajectories in the replay buffer is ordered, this will shuffle the order.
            tool_idxes = tool_idxes.reshape(-1, self.horizon)
            perm = np.random.permutation(len(tool_idxes))
            tool_idxes = tool_idxes[perm].flatten()
            self.tool_idxes[tid] = tool_idxes
        
        B = self.args.batch_size
        n = len(self.tool_idxes[0]) - len(self.tool_idxes[0])%B 
        epoch_tool_idxes = np.array_split(np.random.permutation(self.tool_idxes[tid])[:n], n // B) 
        return epoch_tool_idxes

    def sample_goal_obs(self, obs_idx, hindsight_goal_ratio, device, pcl=False):
        n = len(obs_idx)
        horizon = self.horizon

        traj_id = obs_idx // horizon
        traj_t = obs_idx % horizon
        init_v, target_v = self.buffer['init_v'][obs_idx], self.buffer['target_v'][obs_idx]
        hindsight_future_idx = batch_rand_int(traj_t, horizon, n)
        optimal_step = self.get_optimal_step()  # TODO check this: correctness and efficiency
        hindsight_flag = np.round(np.random.random(n) < hindsight_goal_ratio).astype(np.int32)
        if pcl == '':
            hindsight_goal_imgs = self.buffer['obses'][hindsight_future_idx + traj_id * horizon]
            target_goal_imgs = self.np_target_imgs[target_v]
            goal_obs = img_to_tensor(hindsight_flag[:, None, None, None] * hindsight_goal_imgs +
                                    (1 - hindsight_flag[:, None, None, None]) * target_goal_imgs, mode=self.args.img_mode).to(device, non_blocking=True)
        elif pcl == 'full_pcl':
            hindsight_goal_states = self.buffer['states'][hindsight_future_idx + traj_id * horizon]
            hindsight_goal_states = hindsight_goal_states[:, :3000].reshape(n, -1, 3)
            target_goal_grid = self.np_target_mass_grids[target_v]
            goal_obs = torch.FloatTensor(hindsight_flag[:, None, None] * hindsight_goal_states + ## replace target goal imgs by target point cloud
                                    (1 - hindsight_flag[:, None, None]) * target_goal_grid).to(device, non_blocking=True)
        else:
            raise NotImplementedError
        hindsight_done_flag = (hindsight_future_idx == 0).astype(np.int32)
        target_done_flag = np.logical_or(traj_t == optimal_step[traj_id, traj_t], traj_t % horizon == 0).astype(np.int32)
        done_flag = hindsight_flag * hindsight_done_flag + (1 - hindsight_flag) * target_done_flag
        done_flag = torch.FloatTensor(done_flag).to(device, non_blocking=True)
        hindsight_flag = torch.FloatTensor(hindsight_flag).to(device, non_blocking=True)
        goal_obs = torch.cat([goal_obs for _ in range(self.args.frame_stack)], dim=-1)
        return goal_obs, done_flag, hindsight_flag

    def sample_goals_pcl(self, obs_idx, device):
        ## sample goals for partial point cloud

        n = len(obs_idx)
        horizon = self.horizon
        max_pt = self.max_pts['goal_pcl']
        traj_id = obs_idx // horizon
        traj_t = obs_idx % horizon
        init_v, target_v = self.buffer['init_v'][obs_idx], self.buffer['target_v'][obs_idx]
        optimal_step = self.get_optimal_step()
        
        goal_obs = self.buffer['goal_pcl'][obs_idx]
        goal_pcl_len = self.buffer['goal_pcl_len'][obs_idx]
        goal_obs = torch.FloatTensor(goal_obs).to(device, non_blocking=True)
        goal_pcl_len = torch.tensor(goal_pcl_len).to(device, non_blocking=True)
        done_flag = np.logical_or(traj_t == optimal_step[traj_id, traj_t], traj_t % horizon == 0).astype(np.int32)
        done_flag = torch.FloatTensor(done_flag).to(device, non_blocking=True)
        return goal_obs, goal_pcl_len, done_flag

    def sample_positive_idx(self, obs_idx):
        n = len(obs_idx)
        horizon = self.horizon
        traj_id = obs_idx // horizon
        traj_t = obs_idx % horizon
        future_idx = batch_rand_int(traj_t, horizon, n) + traj_id * horizon
        return future_idx

    def sample_negative_idx(self, obs_idx, epoch):
        horizon = self.horizon
        # Number of trajectories used in the first few epochs, which can be used for negative sampling
        num_traj = min(self.args.step_per_epoch * epoch + self.args.step_warmup, len(self)) // horizon
        assert num_traj > 1
        n = len(obs_idx)
        traj_id = obs_idx // horizon
        neg_traj_id = (traj_id + np.random.randint(1, num_traj)) % num_traj
        traj_t = np.random.randint(0, horizon, n)
        neg_idx = neg_traj_id * horizon + traj_t
        return neg_idx

    def sample_reset_imgs(self, obs_idx):
        horizon = self.horizon
        traj_id = obs_idx // horizon
        reset_lens = self.buffer['reset_motion_lens'][traj_id]

        did_reset_idx = np.where(reset_lens > 0)
        traj_id = traj_id[np.where(reset_lens > 0)]
        reset_lens = reset_lens[np.where(reset_lens > 0)]
        reset_idx = batch_rand_int(0, reset_lens, len(reset_lens))
        reset_imgs = self.buffer['reset_motion_obses'][traj_id, reset_idx]
        return did_reset_idx, reset_imgs

    def compute_stats(self, idxes, device):
        full_states = self.buffer['states'][idxes, :3300].reshape(-1, 3)
        full_goals = self.np_target_mass_grids.reshape(-1, 3)
        full_points = np.concatenate([full_states, full_goals], axis=0)
        state_mean, state_std = np.mean(full_points, axis=0), np.std(full_points, axis=0)
        action_mean, action_std = np.mean(self.buffer['actions'][idxes], axis=0), np.std(self.buffer['actions'][idxes], axis=0)
        action_std[4] = 1
        action_std[5] = 1
        y_max, y_min = np.max(self.buffer['actions'][idxes], axis=0), np.min(self.buffer['actions'][idxes], axis=0)
        buf = 0.05* (y_max - y_min)
        y_max, y_min = np.clip(y_max+buf, -1, 1), np.clip(y_min-buf, -1, 1)
        self.stats = {'state_mean':torch.FloatTensor(state_mean).to(device), \
                             'state_std':torch.FloatTensor(state_std).to(device),
                             'action_mean': torch.FloatTensor(action_mean).to(device),
                             'action_std': torch.FloatTensor(action_std).to(device),
                             'y_max':torch.FloatTensor(y_max).to(device),
                             'y_min':torch.FloatTensor(y_min).to(device)}
        from chester import logger
        logger.log(self.stats)
                            

    def sample_tool_transitions(self, batch_tool_idxes, epoch, device):
        """
        :param batch_tool_idxes: A list of index for one mini-batch for each tool
        :return: Dictionary of data in torch tensor, with each item being a list of the corresponding data for each tool
        """
        ret = {}
        for key in ['obses', 'goal_obses', 'actions', 'dones', 'succ_goals', 'succ_labels', 'score_labels', 'hindsight_flags']:
            ret[key] = []

        for tid, curr_tool_idx in enumerate(batch_tool_idxes):
            # For BC
            obs = img_to_tensor(self.sample_stacked_obs(curr_tool_idx, self.args.frame_stack), mode=self.args.img_mode).to(device, non_blocking=True)
            action = torch.FloatTensor(self.buffer['actions'][curr_tool_idx]).to(device, non_blocking=True)
            goal_obs, done, hindsight_flag = self.sample_goal_obs(curr_tool_idx, self.args.hindsight_goal_ratio, device)

            # For succ predictor
            num_pos = int(self.args.pos_ratio * len(curr_tool_idx))
            num_neg = len(curr_tool_idx) - num_pos
            num_pos_reset = int(self.args.pos_reset_ratio * num_pos)
            pos_idx = self.sample_positive_idx(curr_tool_idx[:num_pos])  # Include both pos and reset idx
            neg_idx = self.sample_negative_idx(curr_tool_idx[num_pos:], epoch=epoch)

            pos_img = self.buffer['obses'][pos_idx].copy()  # Need cloning since they are changed below
            did_reset_idx, reset_imgs = self.sample_reset_imgs(curr_tool_idx[:num_pos_reset])
            pos_img[did_reset_idx] = reset_imgs
            neg_img = self.buffer['obses'][neg_idx]
            succ_goals = img_to_tensor(np.vstack([pos_img, neg_img]), self.args.img_mode).to(device, non_blocking=True)
            succ_label = torch.cat([torch.ones(size=(num_pos,), device=device, dtype=torch.int),
                                    torch.zeros(size=(num_neg,), device=device, dtype=torch.int)])
            score_label = -torch.FloatTensor(self.buffer['info_emds'][curr_tool_idx]).to(device,
                                                                                         non_blocking=True)  # Negative of the EMD as the score

            ret['obses'].append(obs)
            ret['goal_obses'].append(goal_obs)
            ret['succ_goals'].append(succ_goals)
            ret['actions'].append(action)
            ret['dones'].append(done)
            ret['succ_labels'].append(succ_label)
            ret['score_labels'].append(score_label)
            ret['hindsight_flags'].append(hindsight_flag)  # Scores labels come form f(o_curr, o_g) and do not apply to hindsight goals
        return ret
    

    def sample_tool_transitions_bc(self, batch_tool_idxes, epoch, device, pcl=''):
        """
        :param batch_tool_idxes: A list of index for one mini-batch for each tool
        :param pcl: If pcl==True, then the returned obs will be state information containing point clouds
        :return: Dictionary of data in torch tensor, with each item being a list of the corresponding data for each tool
        """
        ret = {}
        for key in ['obses', 'goal_obses', 'actions', 'dones', 'obses_pcl_len', 'goals_pcl_len', 'tool_flow']:
            ret[key] = []

        for tid, curr_tool_idx in enumerate(batch_tool_idxes):
            # For BC
            if pcl == '':
                obs = img_to_tensor(self.sample_stacked_obs(curr_tool_idx, self.args.frame_stack), mode=self.args.img_mode).to(device, non_blocking=True)
                goal_obs, done, hindsight_flag = self.sample_goal_obs(curr_tool_idx, self.args.hindsight_goal_ratio, device)
            elif pcl=='full_pcl':
                obs = torch.FloatTensor(self.sample_stacked_obs(curr_tool_idx, self.args.frame_stack, pcl=pcl)).to(device, non_blocking=True)
                goal_obs, done, hindsight_flag = self.sample_goal_obs(curr_tool_idx, self.args.hindsight_goal_ratio, device, pcl=pcl)
            elif pcl=='partial_pcl':
                ## TODO: remove the if-else case to merge full pcl and partial pcl dataloading
                obs, obs_len = self.sample_stacked_obs(curr_tool_idx, self.args.frame_stack, pcl=pcl)
                obs =torch.FloatTensor(obs).to(device, non_blocking=True) # B x (pad_dough + pad_tool) x 3
                obs_len = torch.tensor(obs_len).to(device, non_blocking=True) # B x 2
                goal_obs, goal_pcl_len, done = self.sample_goals_pcl(curr_tool_idx, device)
            else:
                raise NotImplementedError
            action = torch.FloatTensor(self.buffer['actions'][curr_tool_idx]).to(device, non_blocking=True)
    
            
            ret['obses'].append(obs)
            ret['goal_obses'].append(goal_obs)
            ret['actions'].append(action)
            ret['dones'].append(done)
            if pcl=='partial_pcl':
                ret['obses_pcl_len'].append(obs_len)
                ret['goals_pcl_len'].append(goal_pcl_len)
            if self.args.frame == 'tool':
                ret['tool_xyz'] = []
                tool_xyz = torch.FloatTensor(self.buffer['states'][curr_tool_idx, 3300:3303]).to(device, non_blocking=True)
                ret['tool_xyz'].append(tool_xyz) # B x 3
            if self.args.actor_type == 'PointActorToolParticle':
                next_idx = np.clip(curr_tool_idx + 1, 0, self.cur_size)
                tool_flow = (1- done.cpu().detach().numpy())[:,None] * (self.buffer['states'][next_idx, 3000:3300] - self.buffer['states'][curr_tool_idx, 3000:3300])
                tool_flow = torch.FloatTensor(tool_flow.reshape(next_idx.shape[0], -1, 3)).to(device, non_blocking=True)
                ret['tool_flow'].append(tool_flow)
        return ret

    def sample_transition_openloop(self, batch_traj_idxes, device):
        ret = {}
        for key in ['obses', 'goal_obses', 'actions', 'dones', 'obses_pcl_len', 'goals_pcl_len']:
            ret[key] = []
        
        batch_idxes = self.horizon * batch_traj_idxes
        obs, obs_len = self.sample_stacked_obs(batch_idxes, self.args.frame_stack, pcl='partial_pcl')
        obs =torch.FloatTensor(obs).to(device, non_blocking=True) # B x (pad_dough + pad_tool) x 3
        obs_len = torch.tensor(obs_len).to(device, non_blocking=True) # B x 2
        goal_obs, goal_pcl_len, done = self.sample_goals_pcl(batch_idxes, device)
        action_idxes = (batch_traj_idxes.reshape(-1, 1) * self.horizon + np.arange(self.horizon).reshape(1, -1)).flatten()
        action = torch.FloatTensor(self.buffer['actions'][action_idxes].reshape(batch_idxes.shape[0], -1)).to(device, non_blocking=True)
        ret['obses'].append(obs)
        ret['goal_obses'].append(goal_obs)
        ret['actions'].append(action)
        ret['dones'].append(done)
        ret['obses_pcl_len'].append(obs_len)
        ret['goals_pcl_len'].append(goal_pcl_len)
        return ret

