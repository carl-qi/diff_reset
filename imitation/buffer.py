import numpy as np
import pickle
from numpy.core.shape_base import stack
import torch
import gzip
import os
from glob import glob

from torch._C import dtype
from imitation.utils import batch_rand_int, get_partial_pcl, load_target_imgs


class ReplayBuffer(object):
    def __init__(self, args, maxlen=int(150000), her_args=None):
        self.args = args
        self.maxlen = maxlen

        self.cur_size = 0
        self.cur_pt = 0
        self.buffer = {}  # a dictionary of keys, each is an array of size N x dim
        # self.init_vs, self.target_vs, self.action_masks = [], [], []
        self.horizon = 50 if not hasattr(self.args, 'buffer_horizon') else self.args.buffer_horizon
        self.maxtraj = maxlen // self.horizon
        self.her_args = her_args
        if her_args is not None:
            self.reward_fn = her_args.reward_fn
        
        self.max_pts = {
            'tool_pcl': 100 if self.args and self.args.gt_tool else 200,
            'dough_pcl': 1000,
            'goal_pcl': 1000
        }

    def generate_train_eval_split2(self, eval_train=False):
        num_traj = len(self) // self.horizon
        all_idxes = np.arange(num_traj)
        traj_idxes = np.random.permutation(all_idxes)
        # self.hard_coded_eval_idxes = np.array([0,1,2,3,4])
        self.hard_coded_eval_idxes = np.array([4, 20, 28, 35, 47, 52, 67, 83, 86, 89, 95, 118, 120, 122, 125, 127, 128, 130, 131, 134, 140, 146]) # 0214 official experiments

        self.eval_traj_idx = self.hard_coded_eval_idxes
        if eval_train:
            self.train_traj_idx = traj_idxes
        else:
            self.train_traj_idx= np.array([i for i in traj_idxes if i not in self.hard_coded_eval_idxes])
        
        self.train_idx = (self.train_traj_idx.reshape(-1, 1) * self.horizon + np.arange(self.horizon).reshape(1, -1)).flatten()
        self.eval_idx = (self.eval_traj_idx.reshape(-1, 1) * self.horizon + np.arange(self.horizon).reshape(1, -1)).flatten()
        print("number of training trajectories: {0}, transitions: {1}".format(self.train_traj_idx.shape[0], self.train_idx.shape[0]))
        print("train_idxes:", self.train_traj_idx)
        print("eval_idxes:", self.hard_coded_eval_idxes)


    def generate_train_eval_split(self, train_ratio=0.9, traj_limit=-1, eval_train=False, filter=True):
        if eval_train:
            train_ratio = 1
        num_traj = len(self) // self.horizon
        traj_limit = num_traj if traj_limit == -1 else traj_limit
        all_idxes = np.arange(num_traj)
        mid_thres = self.args.mid_thres
        if filter:
            all_emds = self.buffer['info_emds'][:self.cur_size].reshape(-1, self.horizon)
            mid_normalized_emds = (all_emds[:,0] - all_emds[:, 50]) / all_emds[:, 0]
            final_normalized_emds = (all_emds[:, 0] - all_emds[:, -1]) / all_emds[:, 0]
            idx = np.where((final_normalized_emds > 0.3)*(mid_normalized_emds > mid_thres))
            all_idxes = all_idxes[idx]  
        traj_idxes = np.random.permutation(all_idxes)

        num_train_traj = int(len(traj_idxes) * train_ratio)
        assert not hasattr(self, 'train_traj_idx')
        self.train_traj_idx, self.eval_traj_idx = traj_idxes[:num_train_traj][:traj_limit], traj_idxes[num_train_traj:]
        if eval_train:
            if traj_limit == 1:
                # self.train_traj_idx = np.array([35])
                # self.hard_coded_eval_idxes = np.array([35])
                self.train_traj_idx = np.array([97])
                self.hard_coded_eval_idxes = np.array([97])
            elif traj_limit == num_traj:
                if mid_thres == 0.15:
                    # self.hard_coded_eval_idxes = np.array([  0,  93, 147, 100,  97,  96, 126, 158, 102, 119])
                    self.hard_coded_eval_idxes = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) # 0.6_0.15_small_tool
                elif mid_thres == 0.3:
                    # self.hard_coded_eval_idxes = np.array([ 21,  22, 191,  25,  82,  86, 151, 199,  46,  77])
                    self.hard_coded_eval_idxes = np.array([ 97,  82,  81,  18,  76, 115, 124,  99,  43,  14])
            self.eval_traj_idx = self.hard_coded_eval_idxes

        elif traj_limit == num_traj:
            if mid_thres == 0.15:
                # self.hard_coded_eval_idxes = np.array([ 17,  34,  92,  19, 133,  86, 168, 148, 175, 158])
                self.hard_coded_eval_idxes = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) # 0.6_0.15_small_tool
            elif mid_thres == 0.3:
                # self.hard_coded_eval_idxes = np.array([15,  24, 35, 129,  82,  77])
                self.hard_coded_eval_idxes = np.array([ 97,  82,  81,  18,  76, 115, 124,  99,  43,  14]) # small_tool_dataset
            self.train_traj_idx = np.array([i for i in traj_idxes if i not in self.hard_coded_eval_idxes])
            self.eval_traj_idx = self.hard_coded_eval_idxes

        self.train_idx = (self.train_traj_idx.reshape(-1, 1) * self.horizon + np.arange(self.horizon).reshape(1, -1)).flatten()
        self.eval_idx = (self.eval_traj_idx.reshape(-1, 1) * self.horizon + np.arange(self.horizon).reshape(1, -1)).flatten()
        print("number of training trajectories: {0}, transitions: {1}".format(self.train_traj_idx.shape[0], self.train_idx.shape[0]))
        print("train_idxes:", self.train_traj_idx)
        print("eval_idxes:", self.hard_coded_eval_idxes)
        
    def add(self, traj, save_state=True):
        if len(self.buffer) == 0:
            # initialize buffer
            for key in traj:
                if key == 'states':
                    if save_state:
                        self.buffer[key] = np.empty(shape=(self.maxlen, *traj[key].shape[1:]), dtype=np.float)
                    continue
                elif key in ['init_v', 'target_v']:
                    self.buffer[key] = np.empty(shape=(self.maxlen), dtype=np.int32)
                    continue
                elif key == 'action_mask' and len(np.array(traj[key]).shape) != 2:
                    self.buffer[key] = np.empty(shape=(self.maxlen, np.array(traj[key]).reshape(-1, 1).shape[0]), dtype=np.int32)
                    continue
                elif key == 'mass_grid':
                    # Only save the first few for mass_grid due to memory limit
                    self.buffer[key] = np.empty(shape=(15000, *traj[key].shape[1:]), dtype=traj[key].dtype)
                    continue
                elif not isinstance(traj[key], np.ndarray) or np.prod(traj[key].shape) == 1:  # Do not save scaler value in the buffer
                    continue
                # print(key, traj[key].shape)
                elif key == 'ious':
                    T = traj['ious'].shape[0]
                    self.buffer['ious'] = np.empty(shape=(self.maxlen, T), dtype=traj[key].dtype)
                elif 'reset' in key:
                    self.buffer[key] = np.empty(shape=(self.maxlen//self.horizon, *traj[key].shape[1:]), dtype=traj[key].dtype)
                else:
                    # print('key:', key, traj[key].shape[1:])
                    self.buffer[key] = np.empty(shape=(self.maxlen, *traj[key].shape[1:]), dtype=traj[key].dtype)
        N = traj['actions'].shape[0]

        if self.cur_size + N < self.maxlen:
            idxes = np.arange(self.cur_size, self.cur_size + N)
            reset_idx = np.arange(self.cur_size // self.horizon, (self.cur_size + N) // self.horizon)
            self.cur_size += N
            self.cur_pt = self.cur_size
        else:  # full
            idxes = np.arange(self.cur_pt, self.cur_pt + N) % self.maxlen
            if 'reset_motion_obses' in self.buffer:
                reset_idx = np.arange(self.cur_size // self.horizon, (self.cur_size + N) // self.horizon) % self.buffer['reset_motion_obses'].shape[0]
            self.cur_size = self.maxlen
            self.cur_pt = (self.cur_pt + N) % self.maxlen

        if 'reset_motion_obses' in self.buffer:
            self.buffer['reset_motion_obses'][reset_idx] = traj['reset_motion_obses']
            self.buffer['reset_motion_lens'][reset_idx] = traj['reset_motion_lens']
            self.buffer['reset_info_emds'][reset_idx] = traj['reset_info_emds']
        if 'states' in self.buffer:
            self.buffer['states'][idxes] = traj['states'][:-1]
        if 'dough_pcl' in traj:
            self.buffer['dough_pcl'][idxes] = traj['dough_pcl'][:-1]
            self.buffer['dough_pcl_len'][idxes] = traj['dough_pcl_len'][:-1]
            self.buffer['tool_pcl'][idxes] = traj['tool_pcl'][:-1]
            self.buffer['tool_pcl_len'][idxes] = traj['tool_pcl_len'][:-1]
            self.buffer['goal_pcl'][idxes] = traj['goal_pcl'][:-1]
            self.buffer['goal_pcl_len'][idxes] = traj['goal_pcl_len'][:-1]
        if 'obses' in self.buffer:
            self.buffer['obses'][idxes] = traj['obses'][:-1]
        self.buffer['actions'][idxes] = traj['actions']
        self.buffer['rewards'][idxes] = traj['rewards']
        # self.buffer['mass_grid'][idxes] = traj['mass_grid'][:-1]
        # self.buffer['ious'][idxes] = traj['ious']  # Already remove the last one when computing the pairwise iou
        # self.buffer['target_ious'][idxes] = traj['target_ious'][:-1]
        if 'info_emds' in self.buffer:
            self.buffer['info_emds'][idxes] = traj['info_emds'][:-1]
        if 'info_normalized_performance' in self.buffer:
            self.buffer['info_normalized_performance'][idxes] = traj['info_normalized_performance'][:-1]
        self.buffer['init_v'][idxes] = traj['init_v']
        self.buffer['target_v'][idxes] = traj['target_v']
        self.buffer['action_mask'][idxes] = traj['action_mask'][None]

    def sample(self, B):
        idx = np.random.randint(0, self.cur_size, B)
        batch = {}
        for key in self.buffer:
            batch[key] = self.buffer[key][idx]
        return batch

    def get_goal_obs(self, target_v):
        """ Get goal obs from target_v"""
        if not hasattr(self, 'np_target_imgs'):
            self.np_target_imgs = load_target_imgs(self.args.cached_state_path, ret_tensor=False)
        return self.np_target_imgs[target_v]

    # For TD3
    def her_sample(self, batch_size):
        # First randomply select a batch of transitions. Then with probability future_p, the goals will be replaced with the achieved goals
        # and the rewards will be recomputed
        future_p = 1 - (1. / (self.her_args.replay_k + 1))
        her_bool = (np.random.random(batch_size) < future_p).astype(np.int)
        T = self.horizon
        idx = np.random.randint(0, self.cur_size, batch_size)
        traj_idx, traj_t = idx // T, idx % T

        future_idx = batch_rand_int(traj_t, T, batch_size) + traj_idx * T
        next_idx = traj_idx * T + np.minimum(traj_t + 1, T - 1)
        next_idx = np.minimum(next_idx, self.maxlen - 1)
        not_done = traj_t < T - 1

        if self.args.use_pcl == 'full_pcl':
            obs = self.buffer['states'][idx, :3000].reshape(-1, 3)
            next_obs = self.buffer['states'][next_idx, :3300].reshape(-1, 3)
            real_goal_obs = self.np_target_mass_grids[self.buffer['target_v'][idx]]
            her_goal_obs = self.buffer['states'][future_idx]
            goal_obs = (1-her_bool)[:, None, None] * real_goal_obs + her_bool[:, None, None] * her_goal_obs
        elif self.args.use_pcl == 'partial_pcl':
            dough_pcl, dough_pcl_len, tool_pcl, tool_pcl_len = self.buffer['dough_pcl'][idx], self.buffer['dough_pcl_len'][idx], self.buffer['tool_pcl'][idx], self.buffer['tool_pcl_len'][idx]
            obs = np.concatenate([dough_pcl, tool_pcl], axis=1)
            obs_len = np.concatenate([dough_pcl_len.reshape(-1, 1), tool_pcl_len.reshape(-1, 1)], axis=1)
            goal_obs, goal_len = self.buffer['goal_pcl'][idx], self.buffer['goal_pcl_len'][idx].reshape(-1, 1)
            n_dough_pcl, n_dough_pcl_len, n_tool_pcl, n_tool_pcl_len = self.buffer['dough_pcl'][next_idx], self.buffer['dough_pcl_len'][next_idx], self.buffer['tool_pcl'][next_idx], self.buffer['tool_pcl_len'][next_idx]
            n_obs = np.concatenate([n_dough_pcl, n_tool_pcl], axis=1)
            n_obs_len = np.concatenate([n_dough_pcl_len.reshape(-1, 1), n_tool_pcl_len.reshape(-1, 1)], axis=1)
            action = self.buffer['actions'][idx]
            reward = self.buffer['rewards'][idx].copy()
            return obs, obs_len, goal_obs, goal_len, n_obs, n_obs_len, action, reward, not_done
        else:
            obs = self.buffer['obses'][idx]
            next_obs = self.buffer['obses'][next_idx]
            real_goal_obs = self.get_goal_obs(self.buffer['target_v'][idx])
            her_goal_obs = self.buffer['obses'][future_idx]
            goal_obs = (1 - her_bool)[:, None, None, None] * real_goal_obs + her_bool[:, None, None, None] * her_goal_obs
        action = self.buffer['actions'][idx]

        # Computing HER reward
        reward = self.buffer['rewards'][idx].copy()
        if len(idx[her_bool > 0]) > 0:
            achieved_state = self.buffer['states'][idx[her_bool > 0]]
            goal_state = self.buffer['states'][future_idx[her_bool > 0]]
            her_reward = self.reward_fn(achieved_state, goal_state)
            reward[her_bool > 0] = her_reward

        return obs, goal_obs, action, next_obs, reward, not_done

    def sample_stacked_obs(self, idx, frame_stack, pcl=False):
        # frame_stack =1 means no stacking
        padded_step = np.concatenate([np.zeros(shape=frame_stack - 1, dtype=np.int), np.arange(self.horizon)])
        traj_idx = idx // self.horizon
        traj_t = idx % self.horizon
        idxes = np.arange(0, frame_stack).reshape(1, -1) + traj_t.reshape(-1, 1)
        stacked_t = padded_step[idxes]  # B x frame_stack
        stacked_idx = ((traj_idx * self.horizon).reshape(-1, 1) + stacked_t).T  # frame_stack x B
        if not pcl:
            stack_obs = self.buffer['obses'][stacked_idx] # frame_stack x B x 64x64x4
            stack_obs = np.concatenate(stack_obs, axis=-1)
            return stack_obs
        elif pcl == 'full_pcl':
            stack_obs = self.buffer['states'][stacked_idx] # frame_stack x B x 3316
            stack_obs = np.concatenate([stack_obs[:, :, :3000].reshape(frame_stack, idx.shape[0], -1, 3), \
                stack_obs[:, :, 3000:3300].reshape(frame_stack, idx.shape[0], -1, 3)], axis=2) # frame_stack x B x 1100x3
            stack_obs = np.concatenate(stack_obs, axis=-1)
            return stack_obs
        else:
            assert frame_stack == 1, 'can\'t do frame stacks on partial point cloud yet'
            
            stack_obs_dough = self.buffer['dough_pcl'][idx] # B x 1000 x 3
            dough_pcl_len = self.buffer['dough_pcl_len'][idx]
            if self.args.gt_tool:
                stack_obs_tool = self.buffer['states'][idx, 3000:3300].reshape(idx.shape[0], -1 , 3) # B x 100 x 3
                tool_pcl_len = np.ones((idx.shape[0], 1), dtype=dough_pcl_len.dtype) * 100
            else:    
                stack_obs_tool = self.buffer['tool_pcl'][idx]  # B x 200 x 3
                tool_pcl_len = self.buffer['tool_pcl_len'][idx] # B x 1
            stack_obs = np.concatenate([stack_obs_dough, stack_obs_tool], axis=1)
            obs_len = np.concatenate([dough_pcl_len, tool_pcl_len], axis=1)
            return stack_obs, obs_len

    def load(self, data_path, filename='dataset.gz'):
        print('Loading dataset in', data_path)
        if os.path.isfile(data_path):
            # Skip these datasets which have not been finished
            print('Loading dataset from {}'.format(data_path))
            data_path = data_path.replace('pkl', 'xz')
            with gzip.open(data_path, 'rb') as f:
                # self.__dict__ = pickle.load(f) # Does not work well for more datasets
                d = pickle.load(f)

            dataset_buffer = d['buffer']
            N = len(dataset_buffer['obses'])
            if self.cur_pt + N > self.maxlen:
                print('buffer overflows!!!')
                raise NotImplementedError

            for key in dataset_buffer:
                # print(dataset_buffer[key].shape)
                if key == 'mass_grid':
                    print('loading dataset, skipping mass grid')
                    continue
                if key not in self.buffer:
                    if 'reset' in key:
                        self.buffer[key] = np.empty(shape=(self.maxtraj, *dataset_buffer[key].shape[1:]), dtype=dataset_buffer[key].dtype)
                    else:
                        self.buffer[key] = np.empty(shape=(self.maxlen, *dataset_buffer[key].shape[1:]), dtype=dataset_buffer[key].dtype)
                if 'reset' in key:
                    self.buffer[key][self.cur_pt // self.horizon: (self.cur_pt + N) // self.horizon] = dataset_buffer[key]
                else:
                    self.buffer[key][self.cur_pt: self.cur_pt + N] = dataset_buffer[key]
            self.cur_pt += N
            self.cur_size = self.cur_pt

        else:
            datasets = glob(os.path.join(data_path, '**/*dataset*'), recursive=True)
            for dataset in sorted(datasets):
                self.load(dataset)
            # for exp_folder in sorted(exp_folders):
            #     self.load(os.path.join(exp_folder, filename))
            # dataset_files = glob(os.path.join(data_path, 'dataset*.gz'))
            # print(os.path.join(data_path, 'dataset*.gz'))
            # print(dataset_files)
            # for dataset_file in sorted(dataset_files):
            #     self.load(os.path.join(data_path, dataset_file))

    def save(self, data_path, save_mass_grid=False):
        # https://stackoverflow.com/questions/57983431/whats-the-most-space-efficient-way-to-compress-serialized-python-data
        data_path = data_path.replace('pkl', 'gz')  # gzip compressed file

        d = self.__dict__.copy()  # Shallow copy to avoid large memory usage
        # print(d.keys())

        buffer = {}
        for key in self.buffer:
            if 'reset' in key:
                buffer[key] = self.buffer[key][:self.cur_size // self.horizon]
            else:
                buffer[key] = self.buffer[key][:self.cur_size]
        d['buffer'] = buffer

        # if not save_mass_grid:
        #     if 'mass_grid' in d['buffer']:
        #         del d['buffer']['mass_grid']
        with gzip.open(data_path, 'wb') as f:
            print('dumping to ', data_path)
            pickle.dump(d, f, protocol=4)

    def __len__(self):
        return self.cur_size
