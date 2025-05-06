import os,sys
import torch
import numpy as np
import random
import datetime
import pennylane as qml
from torch import optim
from Global import *
from Read_Data.read_data import *

dev = qml.device("default.qubit", wires=N_Qubit)

def Para():
  par1=torch.rand((N_Lay1,N_par1),device=dev2,dtype=Dtype,requires_grad=True)
  par2=torch.rand((N_Lay2,N_par2),device=dev2,dtype=Dtype,requires_grad=True)
  par3=torch.rand((N_par3),device=dev2,dtype=Dtype,requires_grad=True)
  return par1,par2,par3

def Train_Pyt(bfe,img):
  sam=bfe.shape[0]
  n_batch=int(np.ceil(sam/Batch))
  num=np.arange(sam)
  random.shuffle(num)
  batch=Batch
  for i in range (n_batch): # n_batch
    beg_frm=i*Batch
    end_frm=(i+1)*Batch
    if end_frm>sam:
      end_frm=sam
      batch=end_frm-beg_frm
    bfe2=bfe[num[beg_frm:end_frm]]

  #01 2 345678 9 10 11
  #2 9 10 11 0 1 345 678
  #1678 2 9 10 11 0 345
  #0 1   2 3   4 5
  #0 345 1 678 2 9 10 11
  #012 

    img1=img[num[beg_frm:end_frm]].reshape(batch,4,2,64,8).permute(2,4,1,3,0).reshape(16,256*batch) # 2 9 10 11 01345678
    pred=torch.matmul(Con_Unitary(Par1[0],S_Ker1),img1).reshape(2,16,2,8,8,batch).permute(2,4,0,1,3,5).reshape(16,256*batch) # 1 678 2 9 10 11 0345
    pred=torch.matmul(Con_Unitary(Par1[1],S_Ker1),pred).reshape(256,16,batch).permute(1,0,2).reshape(16,256*batch) # 0345 167829 10 11
    pred=torch.matmul(Con_Unitary(Par1[2],S_Ker1),pred).reshape(2,8,2,8,2,8,batch).permute(0,2,4,1,3,5,6).reshape(8,512*batch) # 012
    pred=torch.matmul(Con_Unitary(Par2[0],S_Ker2),pred).reshape(2,2048,batch).permute(2,0,1)
    pred=pred.pow(2).sum(dim=2)
    err=torch.matmul(pred,Par3)-bfe2
    loss=torch.square(err).sum()
    Opt.zero_grad()
    loss.backward()
    Opt.step()

def Pred_Pyt(img,par1,par2,par3):
  E=[]
  sam=img.shape[0]
  n_batch=int(np.ceil(sam/Batch))
  num=np.arange(sam)
  batch=Batch
  cm0=Con_Unitary(par1[0],S_Ker1)
  cm1=Con_Unitary(par1[1],S_Ker1)
  cm2=Con_Unitary(par1[2],S_Ker1)
  cm3=Con_Unitary(par2[0],S_Ker2)
  for i in range (n_batch):
    beg_frm=i*Batch
    end_frm=(i+1)*Batch
    if end_frm>sam:
      end_frm=sam
      batch=end_frm-beg_frm
    img1=img[num[beg_frm:end_frm]].reshape(batch,4,2,64,8).permute(2,4,1,3,0).reshape(16,256*batch) # 2 9 10 11 01345678
    pred=torch.matmul(cm0,img1).reshape(2,16,2,8,8,batch).permute(2,4,0,1,3,5).reshape(16,256*batch) # 1 678 2 9 10 11 0345
    pred=torch.matmul(cm1,pred).reshape(256,16,batch).permute(1,0,2).reshape(16,256*batch) # 0345 167829 10 11
    pred=torch.matmul(cm2,pred).reshape(2,8,2,8,2,8,batch).permute(0,2,4,1,3,5,6).reshape(8,512*batch) # 012
    pred=torch.matmul(cm3,pred).reshape(2,2048,batch).permute(2,0,1)
    pred=pred.pow(2).sum(dim=2)
    pred=torch.matmul(pred,par3).to('cpu').tolist()
    E=E+pred
  del cm0,cm1,cm2,cm3
  return E

@qml.qnode(dev,interface="torch")
def quantum_circuit(state,cm0,cm1,cm2,cm3):
  qml.QubitStateVector(state,wires=list(range(N_Qubit)))
  qml.QubitUnitary(cm0,wires=[2,9,10,11])
  qml.QubitUnitary(cm1,wires=[1,6,7,8])
  qml.QubitUnitary(cm2,wires=[0,3,4,5])
  qml.QubitUnitary(cm3,wires=[0,1,2])
  return qml.probs(wires=[0])

def Pred_Pen(img,par1,par2,par3):
  E=[]
  cm0=Con_Unitary(par1[0],S_Ker1).to('cpu').numpy()
  cm1=Con_Unitary(par1[1],S_Ker1).to('cpu').numpy()
  cm2=Con_Unitary(par1[2],S_Ker1).to('cpu').numpy()
  cm3=Con_Unitary(par2[0],S_Ker2).to('cpu').numpy()
  for i in range(img.shape[0]):
    pred=quantum_circuit(img[i].to('cpu').numpy(),cm0,cm1,cm2,cm3)
    pred=pred.clone()
    pred=torch.matmul(pred,par3).to('cpu')
    E.append(pred)
  del cm0,cm1,cm2,cm3
  return E

def Main():
  global Par1,Par2,Par3,Opt
  Par1,Par2,Par3=Para()
  Pdb_gen,Pdb_ref,Bfe_gen,Bfe_ref=Read_PDB()
  Img_gen=Read_Img(Pdb_gen)
  Img_ref=Read_Img(Pdb_ref)
  Opt=optim.SGD([Par1,Par2,Par3],lr=learning_rate,momentum=0.9)
  fw=open("Result/rmsd","w", buffering=1)
  for i in range (N_Ite):
    Train_Pyt(Bfe_gen,Img_gen)
    if i%10==9:
      E_gen_pyt=Pred_Pyt(Img_gen,Par1.detach(),Par2.detach(),Par3.detach())
      Out_E_diff(fw,i,Bfe_gen,E_gen_pyt)
  E_gen_pyt=Pred_Pyt(Img_gen,Par1.detach(),Par2.detach(),Par3.detach())
  E_gen_pen=Pred_Pen(Img_gen,Par1.detach(),Par2.detach(),Par3.detach().to('cpu'))
  E_ref_pyt=Pred_Pyt(Img_ref,Par1.detach(),Par2.detach(),Par3.detach())
  E_ref_pen=Pred_Pen(Img_ref,Par1.detach(),Par2.detach(),Par3.detach().to('cpu'))
  Out_Energy('E_train',Bfe_gen,E_gen_pyt,E_gen_pen)
  Out_Energy('E_test',Bfe_ref,E_ref_pyt,E_ref_pen)
  Out_E_diff(fw,10000,Bfe_gen.to('cpu'),E_gen_pen)
  Out_E_diff(fw,10000,Bfe_ref.to('cpu'),E_ref_pen)
  OutPara(Par1.detach(),Par2.detach(),Par3.detach())
  fw.close()

start=datetime.datetime.now()
Main()
end=datetime.datetime.now()
print("執行時間：",end-start)

