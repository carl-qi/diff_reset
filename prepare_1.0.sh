source ~/.bashrc
export PATH=~/miniconda3/bin:$PATH
export PYTHONPATH=$PWD:$PWD/taichi_three:$PWD/marching_cubes/build
export PATH=$PWD/taichi_three:$PATH
export LD_LIBRARY_PATH=/home/hanwenq/miniconda3/lib:/usr/local/cuda/lib64/:$LD_LIBRARY_PATH
. activate dough
