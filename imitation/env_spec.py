import numpy as np
from imitation.utils import to_action_mask


def get_threshold(env_name):
    threshold = {'LiftSpread-v1': 0.4, 'GatherMove-v1': 0.65, 'CutRearrange-v1': 0.65, 'RollExp-v1': 0.6}
    return threshold[env_name]


def get_tool_spec(env, env_name):
    if env_name == 'LiftSpread-v1':
        # [use tool1, ith position means loss for encouraging approaching actions for the ith tool],
        # [use tool2],
        # [use tool1 and 2]
        contact_loss_masks = [
            # [0., 0., 0.],  # No loss for encouraging approaching actions
            [1., 0., 0.],
            # [1., 0., 0.]
        ]
        # 0, 1 means just use the second tool's action space
        action_masks = [
            # to_action_mask(env, [0, 1]),
            to_action_mask(env, [1, 0]),
            # to_action_mask(env, [1, 1])
        ]
    elif 'Roll' in env_name:

        contact_loss_masks = [
            [1., 0.]
        ]
        action_masks = [
            to_action_mask(env, 0)
        ]

    elif env_name == 'GatherMove-v1':
        contact_loss_masks = [
            [1., 0., 0.],  # Gripper action
            [0., 0., 0.],  # No loss for encouraging approaching actions
            [1., 0., 0.]
        ]
        action_masks = [
            to_action_mask(env, [1, 0]),
            to_action_mask(env, [0, 1]),
            to_action_mask(env, [1, 1])
        ]
    elif env_name == 'CutRearrange-v1':
        contact_loss_masks = [
            [1., 0., 0.],  # Kinfe
            [0., 1., 0.],
            [1., 1., 0.]
        ]
        action_masks = [
            to_action_mask(env, 0),
            to_action_mask(env, 1),
            to_action_mask(env, [1, 1])
        ]
    else:
        raise NotImplementedError

    return {'contact_loss_masks': contact_loss_masks,
            'action_masks': action_masks}


def set_render_mode(env, env_name, mode='mesh'):
    import taichi as ti
    import tina
    import os
    # cwd = os.getcwd()
    cwd = os.path.dirname(os.path.abspath(__file__))
    asset_path = os.path.join(cwd, '..', 'assets')

    env.taichi_env.renderer.verbose = False
    if mode == 'mesh':
        # Add table
        model = tina.MeshModel(os.path.join(asset_path, 'table/Table_Coffee_RiceChest.obj'), scale=(0.03, 0.03, 0.03))
        material = tina.Diffuse(tina.Texture(ti.imread(os.path.join(asset_path, "table/Table_Coffee_RiceChest/_Wood_Cherry_.jpg"))))
        env.taichi_env.renderer.bind(-1, model, material, init_trans=env.taichi_env.renderer.state2mat([-0.3, -0.55, 1.0, 1., 0., 0., 0.]))  # -0.38

        # Add cutting board
        if env_name == 'LiftSpread-v1':
            # Cutting board
            board_model = tina.MeshModel(os.path.join(asset_path, 'cuttingboard/Cutting_Board.obj'), scale=(0.02, 0.1, 0.02))
            board_material = tina.Diffuse(tina.Texture(ti.imread(os.path.join(asset_path, 'cuttingboard/textures/Cutting_Board_Diffuse.png'))))
            s = env.taichi_env.primitives[2].get_state(0)
            initial_mat = env.taichi_env.renderer.state2mat(s)
            target_mat = env.taichi_env.renderer.state2mat(
                [s[0] - 0.064182 + 0.05, s[1] + 0.02 - 0.06974, s[2], 0.707, 0., 0.707, 0.])  # TODO Does this need to be changed?
            env.taichi_env.renderer.bind(2, board_model, board_material,
                                         init_trans=np.linalg.pinv(initial_mat) @ target_mat)  # object pose @ init_pose ..
        elif 'Roll' in env_name:
            # Cutting board
            board_model = tina.MeshModel(os.path.join(asset_path, 'cuttingboard/Cutting_Board.obj'), scale=(0.05, 0.02, 0.04))
            board_material = tina.Diffuse(tina.Texture(ti.imread(os.path.join(asset_path, 'cuttingboard/textures/Cutting_Board_Diffuse.png'))))
            i = 1
            s = env.taichi_env.primitives[i].get_state(0)
            initial_mat = env.taichi_env.renderer.state2mat(s)
            target_mat = env.taichi_env.renderer.state2mat(
                [s[0]-0.08, 0.02, s[2], 1., 0., 0., 0.])  # TODO Does this need to be changed?
            
            env.taichi_env.renderer.bind(i, board_model, board_material,
                                         init_trans=np.linalg.pinv(initial_mat) @ target_mat)  # object pose @ init_pose ..
        elif env_name == 'GatherMove-v1':
            # Cutting board
            board_model = tina.MeshModel(os.path.join(asset_path, 'cuttingboard/Cutting_Board.obj'), scale=(0.02, 0.04, 0.02))
            board_material = tina.Diffuse(tina.Texture(ti.imread(os.path.join(asset_path, 'cuttingboard/textures/Cutting_Board_Diffuse.png'))))
            s = env.taichi_env.primitives[2].get_state(0)
            initial_mat = env.taichi_env.renderer.state2mat(s)
            target_mat = env.taichi_env.renderer.state2mat(
                [s[0] - 0.064182 + 0.05, s[1] - 0.007896, s[2], 0.707, 0., 0.707, 0.])  # TODO Does this need to be changed?
            env.taichi_env.renderer.bind(2, board_model, board_material,
                                         init_trans=np.linalg.pinv(initial_mat) @ target_mat)  # object pose @ init_pose ..
    elif mode == 'primitive':
        env.taichi_env.renderer.unbind_all()
    else:
        raise NotImplementedError
