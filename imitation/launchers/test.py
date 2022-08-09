import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
import torch
print(torch.cuda.device_count())