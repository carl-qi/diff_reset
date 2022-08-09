import torch
from torch import nn
import numpy as np
import os
import cv2 as cv
from plb.utils.visualization_utils import save_numpy_as_gif, make_grid
from plb.envs.mp_wrapper import SubprocVecEnv
from scipy.spatial.transform import Rotation as R
# LIGHT_TOOL = (35,0,95)
# DARK_TOOL = (60,1.0,255)
# LIGHT_DOUGH = (0,0,95)
# DARK_DOUGH = (20,1.0,255)
LIGHT_TOOL = (0,0.5,95)
DARK_TOOL = (20,1,255)
LIGHT_DOUGH = (0,0,95)
DARK_DOUGH = (20,0.3,255)

def batch_rand_int(low, high, size):
    # Generate random int from [low, high)
    return np.floor(np.random.random(size) * (high - low) + low).astype(np.int)


env_action_dims = None


def to_action_mask(env, tool_mask):
    global env_action_dims
    if env_action_dims is None:
        if isinstance(env, SubprocVecEnv):
            env_action_dims = env.getattr('taichi_env.primitives.action_dims', idx=0)
        else:
            env_action_dims = env.taichi_env.primitives.action_dims
    action_mask = np.zeros(env_action_dims[-1], dtype=np.float32)
    if isinstance(tool_mask, int):
        tid = tool_mask
        l, r = env_action_dims[tid:tid + 2]
        action_mask[l: r] = 1.
    else:
        for i in range(len(tool_mask)):
            l, r = env_action_dims[i:i + 2]
            action_mask[l: r] = tool_mask[i]
    return action_mask.flatten()


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def gaussian_logl(z):
    """ Loglikelihood of gaussian dist. z: [shape] x z_dim"""
    return -0.5 * torch.sum(z ** 2, dim=-1)


def traj_gaussian_logl(z):
    """z in the shape of num_traj x traj_step x z_dim """
    assert len(z.shape) == 3
    n, num_step, zdim = z.shape
    z_all = gaussian_logl(z.view(n * num_step, zdim)).view(n, num_step)
    return z_all.sum(dim=1)


def aggregate_traj_info(trajs, prefix='info_'):
    infos = {}
    for key in trajs[0].keys():
        if prefix is None or key.startswith(prefix):
            if prefix is None:
                s = key
            else:
                s = key[len(prefix):]
            vals = np.concatenate([np.array(traj[key]).flatten() for traj in trajs])
            infos[f"{s}_mean"] = np.mean(vals)
            # infos[f"{s}_median"] = np.median(vals)
            infos[f"{s}_std"] = np.std(vals)
    return infos


def cv_render(img, name='GoalEnvExt', scale=5):
    '''Take an image in ndarray format and show it with opencv. '''
    img = img[:, :, :3]
    new_img = img[:, :, (2, 1, 0)]
    h, w = new_img.shape[:2]
    new_img = cv.resize(new_img, (w * scale, h * scale))
    cv.imshow(name, new_img)
    cv.waitKey(20)


def debug_show_img(img):
    img = img_to_np(img)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(img[:, :, :3])
    plt.show()


def img_to_tensor(imgs, mode):  # BNMC to BCNM
    imgs = torch.FloatTensor(imgs).permute([0, 3, 1, 2]).contiguous()
    B, C, N, M = imgs.shape
    if mode == 'rgb':
        return imgs.view(B, C // 4, 4, N, M)[:, :, :3, :, :].reshape(B, C // 4 * 3, N, M)
    elif mode == 'rgbd':
        return imgs
    elif mode == 'd':
        return imgs.view(B, C // 4, 4, N, M)[:, :, 3, :, :].reshape(B, C // 4, N, M)


def img_to_np(imgs):  # BCNM to BNMC
    if len(imgs.shape) == 4:
        return imgs.detach().cpu().permute([0, 2, 3, 1]).numpy()
    elif len(imgs.shape) == 3:
        return imgs[None].detach().cpu().permute([0, 2, 3, 1]).numpy()[0]

def get_roller_action_from_transform(r, t):
    action = np.zeros(shape=(1,6))
    action[:, :3] = t.reshape(1, 3)
    rel_rot = R.from_matrix(r)
    ext_euler = rel_rot.as_euler('xyz')
    action[:, 3:] = ext_euler.reshape(1, 3)
    return action

def rigid_transform_3D(A, B):
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t
####### Slower than numpy
# import taichi as ti
#
# m1 = ti.field(dtype=ti.float32, shape=(64, 64, 64), needs_grad=False)
# m2 = ti.field(dtype=ti.float32, shape=(64, 64, 64), needs_grad=False)
#
#
# @ti.kernel
# def iou_kernel() -> ti.float64:
#     dtype = ti.float32
#     ma = ti.cast(0., dtype)
#     mb = ti.cast(0., dtype)
#     I = ti.cast(0., dtype)
#     Ua = ti.cast(0., dtype)
#     Ub = ti.cast(0., dtype)
#     for i in ti.grouped(m1):
#         ti.atomic_max(ma, m1[i])
#         ti.atomic_max(mb, m2[i])
#         I += m1[i] * m2[i]
#         Ua += m1[i]
#         Ub += m2[i]
#     I = I / ma / mb
#     U = Ua / ma + Ub / mb
#     return I / (U - I)
#
#
# def compute_iou(np_m1, np_m2):
#     m1.from_numpy(np_m1)
#     m2.from_numpy(np_m2)
#     return iou_kernel()

def get_iou(a, b, mode='normalized_soft_iou'):
    assert a.shape == b.shape, "shape a: {}, shape  b:{}".format(a.shape, b.shape)
    assert len(a.shape) == 4
    if mode == 'l2':  # Within (0, 1)
        a, b = a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1)
        a = a / np.linalg.norm(a, axis=-1, keepdims=True) ** 2 * np.sum(a, axis=-1)
        b = b / np.linalg.norm(b, axis=-1, keepdims=True) ** 2 * np.sum(b, axis=-1)
        I = np.sum(a * b, axis=-1)
        U = np.sum(a + b, axis=-1) - I
        return I / U
    elif mode == 'soft_iou':
        a = a / np.max(a, axis=(1, 2, 3), keepdims=True)
        b = b / np.max(b, axis=(1, 2, 3), keepdims=True)
        I = np.sum(a * b, axis=(1, 2, 3))
        U = np.sum(a + b, axis=(1, 2, 3)) - I
        return I / U
    elif mode == 'normalized_soft_iou':
        a = a / np.max(a, axis=(1, 2, 3), keepdims=True)
        b = b / np.max(b, axis=(1, 2, 3), keepdims=True)
        s = b.reshape(b.shape[0], -1)
        K = (2 * np.sum(s, axis=-1) - np.sum(s * s, axis=-1)) / (np.sum(s * s, axis=-1))  # normalization
        I = np.sum(a * b, axis=(1, 2, 3))
        U = np.sum(a + b, axis=(1, 2, 3)) - I
        return I / U * K


def compute_pairwise_iou(mass_grid_list):
    m = np.array(mass_grid_list)
    T = m.shape[0]
    m1 = np.tile(m[None, :, :, :, :], [T, 1, 1, 1, 1]).reshape(T * T, 64, 64, 64)
    m2 = np.tile(m[:, None, :, :, :], [1, T, 1, 1, 1]).reshape(T * T, 64, 64, 64)
    ious = get_iou(m1, m2)
    return ious


def load_target_imgs(cached_state_path, mode=None, ret_tensor=True):
    np_target_imgs = np.load(os.path.join(cached_state_path, 'target/target_imgs.npy'))
    if ret_tensor:
        return np_target_imgs, img_to_tensor(np_target_imgs, mode)
    else:
        return np_target_imgs


def load_target_mass_grid(cached_state_path):
    target_mass_grids = []
    import glob
    from natsort import natsorted
    target_paths = natsorted(glob.glob(os.path.join(cached_state_path, 'target/target_[0-9]*.npy')))
    for path in target_paths:
        target_mass_grid = np.load(path)
        # idx = np.random.choice(range(target_mass_grid.shape[0]), 1000, replace=False)
        target_mass_grids.append(target_mass_grid[:1000])
    # import pdb; pdb.set_trace()
    return np.array(target_mass_grids)


def load_target_info(args, device):
    np_target_imgs, target_imgs = load_target_imgs(args.cached_state_path, args.img_mode)
    np_target_mass_grids = load_target_mass_grid(args.cached_state_path)
    target_imgs = target_imgs.to(device)
    target_info = {
        'np_target_imgs': np_target_imgs,
        'target_imgs': target_imgs,
        'np_target_mass_grids': np_target_mass_grids,
    }
    return target_info


def visualize_trajs(trajs, ncol, key, save_name, vis_target=False, demo_obses=None):
    """vis_target: Whether to overlay the target images. demo_obses: whether show original demonstration on the side """
    horizon = max(len(trajs[i]['obses']) for i in range(len(trajs))) + 10  # Add frames after finishing

    all_imgs = []
    for i in range(len(trajs)):
        imgs = []
        for j in range(horizon):
            if j < len(trajs[i]['obses']):
                img = trajs[i]['obses'][j, :, :, :3].copy()  # Do not change the input images
                if vis_target:
                    ratio = trajs[0]['target_img'].shape[0] // img.shape[0]
                    img[:, :, :3] = img[:, :, :3] * 0.7 + trajs[i]['target_img'][::ratio, ::ratio, :3] * 0.3
                if key is not None:
                    if j < len(trajs[i][key]):
                        write_number(img, float(trajs[i][key][j]))
            else:
                # Set the boundary to green
                margin = 2
                img[:margin, :] = img[-margin:, :] = img[:, :margin] = img[:, -margin:] = [0., 1., 0.]
            if demo_obses is not None:
                combined_img = np.hstack([img, demo_obses[i, min(j, len(demo_obses[i]) - 1)]])
                imgs.append(combined_img)
            else:
                imgs.append(img)
        all_imgs.append(imgs)
    a = np.array(all_imgs).swapaxes(0, 1)
    all_frames = []
    for f in range(a.shape[0]):
        frame = make_grid(a[f] * 255, ncol=ncol, padding=3)
        all_frames.append(frame)

    save_numpy_as_gif(np.array(all_frames), save_name)

def write_number_left(img, number, color = (0, 0, 0)):  # Inplace modification
    import cv2
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (1, 60)
    fontScale = 0.4
    fontColor = color
    lineType = 1
    if isinstance(number, str):
        cv2.putText(img, '{}'.format(number),
                    bottomLeftCornerOfText, font,
                    fontScale, fontColor, lineType)
    elif isinstance(number, int):
        cv2.putText(img, str(number),
                    bottomLeftCornerOfText, font,
                    fontScale, fontColor, lineType)
    else:
        cv2.putText(img, '{:.2f}'.format(number),
                    bottomLeftCornerOfText, font,
                    fontScale, fontColor, lineType)
    return img

def write_number(img, number, color = (0, 0, 0)):  # Inplace modification
    import cv2
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (5, 35)
    fontScale = 1
    fontColor = color
    lineType = 1
    thickness = 2
    if isinstance(number, str):
        cv2.putText(img, '{}'.format(number),
                    bottomLeftCornerOfText, font,
                    fontScale, fontColor, thickness, lineType)
    elif isinstance(number, int):
        cv2.putText(img, str(number),
                    bottomLeftCornerOfText, font,
                    fontScale, fontColor, thickness, lineType)
    else:
        cv2.putText(img, '{:.2f}'.format(number),
                    bottomLeftCornerOfText, font,
                    fontScale, fontColor, thickness, lineType)
    return img

def calculate_performance(buffer_path, max_step=50, num_moves=1):
    from imitation.buffer import ReplayBuffer
    buffer = ReplayBuffer(args=None)
    buffer.load(buffer_path)
    horizon = max_step * num_moves
    final_emds = buffer.buffer['info_normalized_performance'][:buffer.cur_size].reshape(-1, horizon)[:, -1]
    print("final_emd_shape:", final_emds.shape)
    return np.mean(final_emds)

def calculate_performance_buffer(buffer, max_step=50, num_moves=1):
    horizon = max_step * num_moves
    final_emds = buffer.buffer['info_normalized_performance'][:buffer.cur_size].reshape(-1, horizon)[:, -1]
    print("final_emd_shape:", final_emds.shape)
    return np.mean(final_emds)

def visualize_dataset(demo_path, cached_state_path, save_name, overlay_target=None, visualize_reset=False, max_step=50, num_moves=1, frequency=1):
    from imitation.buffer import ReplayBuffer
    buffer = ReplayBuffer(args=None)
    buffer.load(demo_path)
    horizon = max_step * num_moves
    N = buffer.cur_size // horizon
    horizon2 = horizon + 10
    print('visualize_dataset, N: ', N)
    all_imgs = []
    for i in range(N):
        if i % frequency == 0:
            img_idx = 0
            # if i % 8 != 0:
            imgs = []
            if overlay_target:
                # init_v = buffer.buffer['init_v'][i * horizon]
                target_v = buffer.buffer['target_v'][i * horizon]
                print('loading img:', os.path.join(cached_state_path, 'target/target_{}.png'.format(target_v)))
                target_img = cv.imread(os.path.join(cached_state_path, 'target/target_{}.png'.format(target_v)))

                target_img = cv.cvtColor(target_img, cv.COLOR_BGR2RGB) / 255.
            # emds = buffer.buffer['info_emds'][i * horizon: (i + 1) * horizon]
            for move in range(num_moves):  
                for j in range(horizon2):
                    if j < horizon:
                        img = buffer.buffer['obses'][i * horizon + max_step * move + j, :, :, :3].copy()
                        num = float(buffer.buffer['info_emds'][i * horizon + max_step * move + j])
                        if overlay_target:
                            img = img[:, :, :3] * 0.8 + target_img * 0.2
                            write_number(img, num)
                            # write_number_left(img, img_idx)
                            img_idx += 1
                    else:
                        margin = 2
                        img[:margin, :] = img[-margin:, :] = img[:, :margin] = img[:, -margin:] = [0., 1., 0.]
                    imgs.append(img)
                if visualize_reset:
                    for j in range(buffer.buffer['reset_motion_obses'].shape[1]):
                        if j < buffer.buffer['reset_motion_lens'][i*num_moves+move]: # If no reset, img will be the last img above
                            img = buffer.buffer['reset_motion_obses'][i*num_moves+move][j].copy()
                            img = img[:, :, :3] * 0.8 + target_img * 0.2
                            reset_info_emd = buffer.buffer['reset_info_emds'][i*num_moves+move][j]
                        elif move == num_moves - 1:
                            margin = 2
                            img[:margin, :] = img[-margin:, :] = img[:, :margin] = img[:, -margin:] = [0., 1., 0.]
                        img_num = write_number(img.copy(), reset_info_emd)
                        # write_number_left(img_num, img_idx)
                        img_idx += 1
                        imgs.append(img_num)
            all_imgs.append(imgs)
    a = np.array(all_imgs).swapaxes(0, 1)
    all_frames = []
    for f in range(a.shape[0]):
        col = min(N, 10)
        frame = make_grid(a[f] * 255, ncol=col)
        all_frames.append(frame)

    save_numpy_as_gif(np.array(all_frames), save_name, fps=20)


def visualize_agent_dataset(buffer, cached_state_path, save_name, overlay_target=None):
    horizon = buffer.horizon
    N = buffer.cur_size // horizon
    print('visualize_dataset, N: ', N)
    all_imgs = []

    for i in range(N):
        # i = N-1
        imgs = []
        if overlay_target:
            # init_v = buffer.buffer['init_v'][i * horizon]
            target_v = buffer.buffer['target_v'][i * horizon]
            target_img = cv.imread(os.path.join(cached_state_path, 'target/target_{}.png'.format(target_v)))
            target_img = cv.cvtColor(target_img, cv.COLOR_BGR2RGB) / 255.
        target_ious = buffer.buffer['target_ious'][i * horizon: (i + 1) * horizon]
        print(buffer.buffer['target_ious'].shape)
        # print('----------------------')
        for j in range(horizon):
            img = buffer.buffer['obses'][i * horizon + j]
            if overlay_target:
                img = img[:, :, :3] * 0.7 + target_img * 0.3
                write_number(img, float(target_ious[j]))
            imgs.append(img)
        all_imgs.append(imgs)
    a = np.array(all_imgs).swapaxes(0, 1)
    all_frames = []
    for f in range(a.shape[0]):
        frame = make_grid(a[f] * 255, ncol=10)
        all_frames.append(frame)

    save_numpy_as_gif(np.array(all_frames), save_name)


from sklearn.neighbors import NearestNeighbors


def chamfer_distance(x, y, metric='l2', direction='bi'):
    """Chamfer distance between two point clouds
    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default ‘l2’
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
    """

    if direction == 'y_to_x':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
    elif direction == 'x_to_y':
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
    elif direction == 'bi':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
    else:
        raise ValueError("Invalid direction type. Supported types: \'y_x\', \'x_y\', \'bi\'")

    return chamfer_dist

def get_camera_params(cfg):
    r, theta, phi, center = cfg.RENDERER['cam_radius'], cfg.RENDERER['cam_theta'], -cfg.RENDERER['cam_phi'], cfg.RENDERER['cam_center']
    camera_pos = np.array(center) + np.array([r*np.cos(phi)*np.sin(theta), r*np.sin(phi), r*np.cos(phi)*np.cos(theta)])
    camera_rot = np.array([theta, phi-np.pi/2,0])
    camera_params = {'pos':camera_pos, 'angle':camera_rot}
    return camera_params

def get_camera_matrix(env):
    if isinstance(env, SubprocVecEnv):
        view, proj = env.getfunc('taichi_env.renderer.scene.control.get_camera', 0)
    else:
        view, proj = env.taichi_env.renderer.scene.control.get_camera()
    return view, proj

def get_partial_pcl2(img, light, dark, view, proj, random_mask=False, p=0.8):
    img_rgb = (img[:,:,:3]*255).astype(np.float32)
    img_hsv = cv.cvtColor(img_rgb, cv.COLOR_RGB2HSV)
    mask = cv.inRange(img_hsv, light, dark)
    img_size=img.shape[0]
    
    us, vs = np.repeat(np.arange(img_size), img_size), np.tile(np.arange(img_size), img_size)
    v2w = np.linalg.inv(proj @ view)
    one = np.ones(us.shape)
    cam_coords = np.stack([us/img_size*2-1, vs/img_size*2-1, img[:,:,-1].flatten(), one], axis=1).T
    res = v2w @ cam_coords
    rew = np.dot(v2w[3:4, :], cam_coords)
    pcl = (res/rew).T

    if random_mask:
        random_mask = np.random.rand(mask.flatten().shape[0]) < p
        pcl = pcl[np.where(np.logical_and(mask.flatten()>0, random_mask))]
    else:
        pcl = pcl[np.where(mask.flatten()>0)]
    return pcl    

def get_partial_pcl(img, light, dark, camera_params, random_mask=False, p=0.8):
    img_rgb = (img[:,:,:3]*255).astype(np.float32)
    img_hsv = cv.cvtColor(img_rgb, cv.COLOR_RGB2HSV)
    mask = cv.inRange(img_hsv, light, dark)
    # result = cv.bitwise_and(img_rgb, img_rgb, mask=mask)
    # img = img*(np.expand_dims(mask,-1) > 0)
    img_size=img.shape[0]
    camera_params.update({'height': img_size, 'width':img_size})
    us, vs = np.repeat(np.arange(img_size), img_size), np.tile(np.arange(img_size), img_size)
    pcl = uv_to_world_pos(us,vs,img[:, :, -1].flatten(), camera_params)
    if random_mask:
        random_mask = np.random.rand(mask.flatten().shape[0]) < p
        pcl = pcl[np.where(np.logical_and(mask.flatten()>0, random_mask))]
    else:
        pcl = pcl[np.where(mask.flatten()>0)]
    return pcl

def intrinsic_from_fov(height, width, fov=90):
    """
    Basic Pinhole Camera Model
    intrinsic params from fov and sensor width and height in pixels
    Returns:
        K:      [4, 4]
    """
    px, py = (width / 2, height / 2)
    hfov = fov / 360. * 2. * np.pi
    fx = width / (2. * np.tan(hfov / 2.))

    vfov = 2. * np.arctan(np.tan(hfov / 2) * height / width)
    fy = height / (2. * np.tan(vfov / 2.))

    return np.array([[fx, 0, px, 0.],
                     [0, fy, py, 0.],
                     [0, 0, 1., 0.],
                     [0., 0., 0., 1.]])


def get_rotation_matrix(angle, axis):
    axis = axis / np.linalg.norm(axis)
    s = np.sin(angle)
    c = np.cos(angle)

    m = np.zeros((4, 4))

    m[0][0] = axis[0] * axis[0] + (1.0 - axis[0] * axis[0]) * c
    m[0][1] = axis[0] * axis[1] * (1.0 - c) - axis[2] * s
    m[0][2] = axis[0] * axis[2] * (1.0 - c) + axis[1] * s
    m[0][3] = 0.0

    m[1][0] = axis[0] * axis[1] * (1.0 - c) + axis[2] * s
    m[1][1] = axis[1] * axis[1] + (1.0 - axis[1] * axis[1]) * c
    m[1][2] = axis[1] * axis[2] * (1.0 - c) - axis[0] * s
    m[1][3] = 0.0

    m[2][0] = axis[0] * axis[2] * (1.0 - c) - axis[1] * s
    m[2][1] = axis[1] * axis[2] * (1.0 - c) + axis[0] * s
    m[2][2] = axis[2] * axis[2] + (1.0 - axis[2] * axis[2]) * c
    m[2][3] = 0.0

    m[3][0] = 0.0
    m[3][1] = 0.0
    m[3][2] = 0.0
    m[3][3] = 1.0

    return m


def get_matrix_world_to_camera(camera_param):
    """
    camera_param is a dictionary in the common format used in softgym:
        {'pos': cam_pos,
         'angle': cam_angle,
         'width': self.camera_width,
         'height': self.camera_height}
    """
    cam_x, cam_y, cam_z = camera_param['pos'][0], camera_param['pos'][1], \
                          camera_param['pos'][2]
    cam_x_angle, cam_y_angle, cam_z_angle = camera_param['angle'][0], \
                                            camera_param['angle'][1], \
                                            camera_param['angle'][2]

    # get rotation matrix: from world to camera
    matrix1 = get_rotation_matrix(- cam_x_angle, [0, 1, 0])
    matrix2 = get_rotation_matrix(- cam_y_angle - np.pi, [1, 0, 0])
    rotation_matrix = matrix2 @ matrix1

    # get translation matrix: from world to camera
    translation_matrix = np.eye(4)
    translation_matrix[0][3] = - cam_x
    translation_matrix[1][3] = - cam_y
    translation_matrix[2][3] = - cam_z

    return rotation_matrix @ translation_matrix


def uv_to_world_pos(u, v, z, camera_params):
    height, width = camera_params['height'], camera_params['width']
    K = intrinsic_from_fov(height, width, fov=60)  # the fov is 90 degrees
    x0 = K[0, 2]
    y0 = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]
    matrix_world_to_camera = get_matrix_world_to_camera(camera_params)
    one = np.ones(u.shape)
    l = fx * np.tan(np.pi / 6)
    z = z * np.cos(np.arctan(np.sqrt(((v-x0)/x0*l)**2 + ((u-y0)/y0*l)**2) / fx)) 
    x = (v - x0) * z / fx
    y = (u - y0) * z / fy
    cam_coords = np.stack([x, y, z, one], axis=1)
    # return cam_coords
    cam2world = np.linalg.inv(matrix_world_to_camera).T
    world_coords = cam_coords @ cam2world
    return world_coords


def world_to_uv(matrix_world_to_camera, world_coordinate, height=360, width=360):
    world_coordinate = np.concatenate([world_coordinate, np.ones((len(world_coordinate), 1))], axis=1)  # n x 4
    camera_coordinate = matrix_world_to_camera @ world_coordinate.T  # 3 x n
    camera_coordinate = camera_coordinate.T  # n x 3
    K = intrinsic_from_fov(height, width, fov=60)  # the fov is 90 degrees

    u0 = K[0, 2]
    v0 = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]

    x, y, depth = camera_coordinate[:, 0], camera_coordinate[:, 1], camera_coordinate[:, 2]
    u = (x * fx / depth + u0).astype("int")
    v = (y * fy / depth + v0).astype("int")

    return u, v

def rotate_y_axis_angle(primitive_states, tid, degree):
    old_state = primitive_states[tid]
    p = R.from_quat([old_state[4], old_state[5], old_state[6], old_state[3]])
    q = R.from_rotvec(degree * np.array([0, 1, 0])) # hard coded rotate 90 around y-axis for now
    quat = R.as_quat(q*p)
    new_state = np.array([old_state[0], old_state[1], old_state[2], quat[-1], quat[0], quat[1], quat[2]])
    new_primitive_states = [s for s in primitive_states]
    new_primitive_states[tid] = new_state
    return new_primitive_states