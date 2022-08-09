import numpy as np

# Used for composing
test_init_v = np.array([95, 95, 95, 91, 91, 91, ])
test_target_v = np.array([25, 195, 128, 21, 191, 128])

## Mostly translation only.
test_init_v = np.array([80, 90, 70, 85, 81, 95, 65])
test_target_v = np.array([20, 30, 10, 25, 21, 25, 25])

# test_init_v = np.array([95])
# test_target_v = np.array([25])

from imitation.sampler import sample_traj
from imitation.utils import visualize_trajs
import os


def test_agent(env, agent, test_init_v, test_target_v, target_info, save_path=None):
    """ Evaluate the agent on the given env configurations. Save the trajectories. Repeat for both tids"""
    target_imgs, np_target_imgs, np_target_mass_grids = target_info['target_imgs'], target_info['np_target_imgs'], target_info['np_target_mass_grids']

    trajs = []
    for tid in [0, 1]:
        for init_v, target_v in zip(test_init_v, test_target_v):
            reset_key = {'init_v': init_v, 'target_v': target_v}
            traj = sample_traj(env, agent, reset_key, tid=tid, compute_ious=False, compute_target_ious=True,
                               target_img=target_imgs[reset_key['target_v']], target_mass_grids=np_target_mass_grids)
            traj['target_img'] = np_target_imgs[reset_key['target_v']]
            trajs.append(traj)

    if save_path is not None:
        if save_path[-4:] != '.gif':
            save_path = os.path.join(save_path, f"test_trajs.gif")
        visualize_trajs(trajs, len(test_init_v), key = 'info_normalized_performance', save_name=save_path, vis_target=True, demo_obses=None)
    return trajs


def test_agent_tid(env, agent, test_init_v, test_target_v, target_info, tids, save_path=None):
    """ Evaluate the agent on the given env configurations. Save the trajectories"""
    target_imgs, np_target_imgs, np_target_mass_grids = target_info['target_imgs'], target_info['np_target_imgs'], target_info['np_target_mass_grids']

    trajs = []
    for init_v, target_v, tid in zip(test_init_v, test_target_v, tids):
        reset_key = {'init_v': init_v, 'target_v': target_v}
        traj = sample_traj(env, agent, reset_key, tid=tid, compute_ious=False, compute_target_ious=True,
                           target_img=target_imgs[reset_key['target_v']], target_mass_grids=np_target_mass_grids)
        traj['target_img'] = np_target_imgs[reset_key['target_v']]
        trajs.append(traj)

    if save_path is not None:
        if save_path[-4:] != '.gif':
            save_path = os.path.join(save_path, f"test_trajs.gif")
        visualize_trajs(trajs, len(test_init_v), key='info_normalized_performance', save_name=save_path, vis_target=True, demo_obses=None)
    return trajs
