import numpy as np
import torch

N_Qubit=12
N_Lay1=2
S_Ker1=32
N_Lay2=1
S_Ker2=16
N_par1=S_Ker1*S_Ker1
N_par2=S_Ker2*S_Ker2
N_par3=2
N_Ite=10000
learning_rate=1e-3
Batch=100

Dtype=torch.float64
dev2="cpu"


