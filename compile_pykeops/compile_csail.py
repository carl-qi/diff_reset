import os

dir = '/data/vision/torralba/scratch/chuang/xingyu/Projects/pykeops_cache/'
os.makedirs(dir, exist_ok=True)
print('Setting pykeops dir to ', dir)

for i in range(10):
    print(i)

    os.system(
        f"CUDA_VISIBLE_DEVICES={i} python -c 'import pykeops; pykeops.set_bin_folder(\"/data/vision/torralba/scratch/chuang/xingyu/Projects/pykeops_cache/\"); pykeops.test_torch_bindings()'")
