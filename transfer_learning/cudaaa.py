import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

np_arry_a = np.arange(500000000).reshape(1, 500000000)
np_arry_b = np.arange(500000000).reshape(1, 500000000)
tensor_a = torch.tensor(np_arry_a, device='cuda')
tensor_b = torch.tensor(np_arry_b, device='cuda')
gpu_1 = time.time()
tensor_a* tensor_b 
gpu_2 = time.time()
print('gpu computation time: ', gpu_2-gpu_1)

cpu_1 = time.time()
np_arry_a* np_arry_b 
cpu_2 = time.time()
print('cpu computation time: ', cpu_2-cpu_1)