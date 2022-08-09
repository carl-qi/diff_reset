import os

p = os.environ.get("CUDA_VISIBLE_DEVICES")
print("Launching a program with cuda ", p)