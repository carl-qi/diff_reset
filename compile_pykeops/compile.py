if __name__ == '__main__':
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    import pykeops
    import torch
    import os

    dir = os.path.join(os.path.expanduser("~"), '.pykeops_cache/'+torch.cuda.get_device_name(0).replace(' ', '_'))

    os.makedirs(dir, exist_ok=True)
    print('Setting pykeops dir to ', dir)
    pykeops.set_bin_folder(dir)
    pykeops.test_torch_bindings()
    from plb.envs.mp_wrapper import make_mp_envs
    env = make_mp_envs('Roll-v1', 1, 0)
    obs = env.reset([{'init_v': 0, 'target_v': 0} for j in range(1)])
    print('reset done')
    print(obs[0].shape)