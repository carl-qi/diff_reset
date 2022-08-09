import pykeops
import torch

dir = '/data/vision/torralba/scratch/chuang/xingyu/Projects/pykeops_cache/'+torch.cuda.get_device_name(0)
os.makedirs(dir, exist_ok=True)
print('Setting pykeops dir to ', dir)
pykeops.set_bin_folder(dir)
pykeops.test_torch_bindings()