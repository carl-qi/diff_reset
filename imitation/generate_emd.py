from imitation.imitation_buffer import ImitationReplayBuffer
from imitation.train_full import get_args
import glob
import os.path as osp
import os
import numpy as np
from tqdm import tqdm
from geomloss import SamplesLoss
import torch

loss_fn = SamplesLoss(loss='sinkhorn', p=1, blur=0.001)


def compute(particle_state, goal_particle_state):
    idx = np.random.choice(range(len(particle_state)), 500)
    particle_state = particle_state[idx]
    idx = np.random.choice(range(len(goal_particle_state)), 500)
    goal_particle_state = goal_particle_state[idx]
    if not isinstance(particle_state, torch.Tensor):
        particle_state = torch.FloatTensor(particle_state).to('cuda', non_blocking=True)
        goal_particle_state = torch.FloatTensor(goal_particle_state).to('cuda', non_blocking=True)

    emd = loss_fn(particle_state, goal_particle_state)
    return emd


args = get_args()


def state_to_dict(vec):
    if len(vec.shape) == 1:
        particles = vec[:1000 * 3].reshape([1000, 3])
    elif len(vec.shape) == 2:
        N = vec.shape[0]
        particles = vec[:, :1000 * 3].reshape([N, 1000, 3])
    return {'particles': particles}


def generate_emd(dataset_path):
    """ Generate emd for the dataset and then save it back"""
    buffer = ImitationReplayBuffer(args)
    buffer.load(dataset_path)
    args.cached_state_path = 'data/hza/buffers/CutRearrange/'
    emds = []
    print('length:', len(buffer))
    for i in tqdm(range(len(buffer))):
        target_v = buffer.buffer['target_v'][i]
        target_state = np.load(os.path.join(args.cached_state_path, f'target/target_{target_v}.npy'))
        curr_state = state_to_dict(buffer.buffer['states'][i])

        emd = compute(curr_state['particles'], target_state)
        emds.append(emd.item())
    np_emds = np.array(emds)
    final_emds = np_emds.reshape(-1, 50)[:, -1]
    buffer.buffer['info_emds'][:len(np_emds)] = np_emds
    buffer.buffer['reset_info_emds'][:len(final_emds), :] = final_emds[:, None]

    print('start saving')
    import time
    st_time = time.time()
    buffer.save(dataset_path)
    print(f'{dataset_path} saved in {time.time() - st_time} s')


def run_task(arg_vv, log_dir, exp_name):  # Chester launch
    from chester import logger
    logger.configure(dir=log_dir, exp_name=exp_name)
    log_dir = logger.get_dir()
    assert log_dir is not None
    os.makedirs(log_dir, exist_ok=True)

    dataset_path = arg_vv['dataset_path']
    print(dataset_path)
    generate_emd(dataset_path)


if __name__ == '__main__':
    dataset_folder = './data/autobot/0829_PushSpread/0829_PushSpread/'
    remote_folder = '/home/xlin3/Projects/PlasticineLab/data/local/0829_PushSpread/'
    all = []
    for exp_folder in sorted(glob.glob(osp.join(dataset_folder, '*/'))):
        dataset_path = osp.join(exp_folder, 'dataset.gz')
        basename = os.path.basename(os.path.dirname(exp_folder))
        remote_path = osp.join(remote_folder, basename, 'dataset.gz')
        all.append(remote_path)
        print('local:', dataset_path)
        print('remote:', remote_path)
        generate_emd(dataset_path)
        cmd = f"scp {dataset_path} autobot:{remote_path} &"
        os.system(cmd)
