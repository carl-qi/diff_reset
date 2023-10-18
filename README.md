# Diff-Reset

## Introduction
This repository contains the official implementation of the following paper:

**(RA-L 2022) Learning Closed-loop Dough Manipulation Using a Differentiable Reset Module**

Carl Qi, Xingyu Lin, David Held

## Usage
1. Install `python3 -m pip install -e .`
2. Install [torch (version 1.9.0 tested)](https://pytorch.org/get-started/previous-versions/)
    * We tested `pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html` on RTX 3090.
3. Install packages for computing the EMD loss:
    * [pykeops (1.5)](https://www.kernel-operations.io/keops/python/installation.html) by
      running `pip install pykeops==1.5`
    * [geomloss](https://www.kernel-operations.io/geomloss/api/install.html) by running `pip install geomloss`
5. Install **chester** from https://github.com/Xingyu-Lin/chester.
6. Run `python imitation/launchers/launch_gen_data.py` to run Diff-Reset.

## Cite

If you find this codebase useful in your research, please consider citing:

```
@ARTICLE{qi2022dough,
  author={Qi, Carl and Lin, Xingyu and Held, David},
  journal={IEEE Robotics and Automation Letters}, 
  title={Learning Closed-Loop Dough Manipulation Using a Differentiable Reset Module}, 
  year={2022},
  volume={7},
  number={4},
  pages={9857-9864},
  doi={10.1109/LRA.2022.3191239}}
```
