PATH=~/carl/miniconda3/bin:$PATH
export PYTHONPATH=$PWD:$PWD/taichi_three
export PATH=$PWD/taichi_three:$PATH:/usr/local/cuda-11.1/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.1/lib64
conda activate dough