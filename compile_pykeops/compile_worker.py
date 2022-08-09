import pykeops
import torch
import os

if '3090' in torch.cuda.get_device_name(0):
    dir = '~/.cache/pykeops_3090/'
    os.makedirs(dir, exist_ok=True)
    print('Setting pykeops dir to ', dir)
    pykeops.set_bin_folder(dir)
elif '1080' in torch.cuda.get_device_name(0):
    dir = '~/.cache/pykeops_1080/'
    os.makedirs(dir, exist_ok=True)
    print('Setting pykeops dir to ', dir)
    pykeops.set_bin_folder(dir)
elif '2080' in torch.cuda.get_device_name(0):
    dir = '~/.cache/pykeops_2080/'
    os.makedirs(dir, exist_ok=True)
    print('Setting pykeops dir to ', dir)
    pykeops.set_bin_folder(dir)
elif '5000' in torch.cuda.get_device_name(0):
    dir = '~/.cache/pykeops_5000/'
    os.makedirs(dir, exist_ok=True)
    print('Setting pykeops dir to ', dir)
    pykeops.set_bin_folder(dir)

pykeops.test_torch_bindings()