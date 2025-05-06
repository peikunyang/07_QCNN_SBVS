import os,sys
import torch
import numpy as np
import random
import datetime
import pennylane as qml
from Global import *
from Read_Data.read_data import *

dev = qml.device("default.mixed", wires=N_Qubit)

@qml.qnode(dev, interface="torch")
def quantum_circuit(state, cm0, cm1, cm2):
    qml.QubitStateVector(state, wires=list(range(N_Qubit)))
    noise_level = 0.05
    phase_damping = 0.03
    qml.QubitUnitary(cm0, wires=[1, 2, 7, 8])
    qml.QubitUnitary(cm1, wires=[0, 1, 5, 6])
    qml.QubitUnitary(cm2, wires=[0, 3, 4])
    qml.DepolarizingChannel(noise_level, wires=0)
    qml.PhaseDamping(phase_damping, wires=0)
    return qml.probs(wires=[0])

def Pred_Pen(img,par1,par2,par3):
  E=[]
  cm0=Con_Unitary(par1[0],S_Ker1).to('cpu').numpy()
  cm1=Con_Unitary(par1[1],S_Ker1).to('cpu').numpy()
  cm2=Con_Unitary(par2[0],S_Ker2).to('cpu').numpy()
  for i in range(img.shape[0]):
    pred=quantum_circuit(img[i].to('cpu').numpy(),cm0,cm1,cm2)
    pred=torch.tensor(pred,dtype=torch.float64).clone()
    pred=torch.matmul(pred,par3).to('cpu')
    E.append(pred)
  del cm0,cm1,cm2
  return E

def Read_Par():
  with open(f'../../../../2_train/1_9Q/2_con_443/test4/Result/par',"r") as f:
    data=[float(num) for line in f for num in line.strip().split()]
  par1=torch.tensor(data[:N_Lay1*N_par1],dtype=torch.float32).reshape(N_Lay1,N_par1)
  par2=torch.tensor(data[N_Lay1*N_par1:N_Lay1*N_par1+N_Lay2*N_par2],dtype=torch.float32).reshape(N_Lay2,N_par2)
  par3=torch.tensor(data[N_Lay1*N_par1+N_Lay2*N_par2:],dtype=torch.float64).reshape(N_par3)
  return par1,par2,par3

def Main():
  global Par1,Par2,Par3
  Par1,Par2,Par3=Read_Par()
  Pdb_gen,Pdb_ref,Bfe_gen,Bfe_ref=Read_PDB()
  Img_gen=Read_Img(Pdb_gen)
  Img_ref=Read_Img(Pdb_ref)
  fw=open("Result/rmsd","w", buffering=1)
  E_gen_pen=Pred_Pen(Img_gen,Par1.detach(),Par2.detach(),Par3.detach().to('cpu'))
  E_ref_pen=Pred_Pen(Img_ref,Par1.detach(),Par2.detach(),Par3.detach().to('cpu'))
  Out_Energy('E_train',Bfe_gen,E_gen_pen)
  Out_Energy('E_test',Bfe_ref,E_ref_pen)
  Out_E_diff(fw,10000,Bfe_gen.to('cpu'),E_gen_pen)
  Out_E_diff(fw,10000,Bfe_ref.to('cpu'),E_ref_pen)
  fw.close()

start=datetime.datetime.now()
Main()
end=datetime.datetime.now()
print("執行時間：",end-start)

