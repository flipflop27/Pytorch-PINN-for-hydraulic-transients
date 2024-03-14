import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from torch.utils.data import Dataset, DataLoader

"""Inputs for training data """

train_data = r"file_path"
df_train_data = pd.read_excel(Training Data.xlsx) 

x_inp = pd.read_excel(train_data, usecols=["Distance, x"])
t_inp = pd.read_excel(train_data, usecols=["Time (s)"]) 
h_inp = pd.read_excel(train_data, usecols=["Head [m]"]) 
q_inp = pd.read_excel(train_data, usecols=["Flowrate [m3/s]"])

x_train = torch.tensor(x_inp.values).to(dtype=torch.float32).view(-1,1)
t_train = torch.tensor(t_inp.values).to(dtype=torch.float32).view(-1,1)
h_train = torch.tensor(h_inp.values).to(dtype=torch.float32).view(-1,1)
q_train = torch.tensor(q_inp.values).to(dtype=torch.float32).view(-1,1)

"""Initial Conditions"""
x_init = torch.tensor(155.).view(-1,1).requires_grad_(True)
t_init = torch.tensor(0.).view(-1,1).requires_grad_(True)

class TrainData(Dataset):
    def __init__(self, x_traindata, t_traindata, h_train, q_train):
        train_input = torch.cat([x_traindata, t_traindata], axis = 1)
        train_aim = torch.cat([h_train, q_train], axis = 1)
        self.train_input = train_input
        self.train_aim = train_aim
        
    def __len__(self):
        return self.train_input.size(0)
    
    def __getitem__(self, idx):
        input_value_x = self.train_input[idx, 0]
        input_value_t = self.train_input[idx, 1] 
        output_value_h = self.train_aim[idx, 0]
        output_value_q = self.train_aim[idx, 1]
        return input_value_x, input_value_t, output_value_h, output_value_q
    
class CollocData(Dataset):
    def __init__(self, x_colloc, t_colloc):
        colloc_input = torch.cat([x_colloc, t_colloc], axis = 1)
        self.colloc_input = colloc_input
        
    def __len__(self):
        return self.colloc_input.size(0)
    
    def __getitem__(self, idx):
        input_x = self.colloc_input[idx, 0]
        input_t = self.colloc_input[idx, 1]
        return input_x, input_t

class FCN(nn.Module):
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.ReLU
        
        self.fcs = nn.Sequential(
            nn.Linear(N_INPUT, N_HIDDEN),
            activation()
        )
        
        self.fch2 = nn.Sequential(
            nn.Linear(N_HIDDEN, N_HIDDEN),
            activation()
            )
                
        self.fch = nn.Sequential(*[
                    nn.Sequential(*[
                        nn.Linear(N_HIDDEN, N_HIDDEN),
                        activation()]) for _ in range(N_LAYERS - 1)])
                
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)  # Output layer
        
        self.bn1 = nn.BatchNorm1d(N_HIDDEN)
        
        self.bn2 = nn.BatchNorm1d(N_OUTPUT)
        
        init.kaiming_normal_(self.fcs[0].weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.fch2[0].weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.fce.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x, t):
        inputs = torch.cat([x, t], axis = 1)
        
        inputs = self.fcs(inputs)
        
        #inputs = self.fch(inputs)
        
        inputs = self.fch2(inputs) #1
        inputs = self.bn1(inputs)
        inputs = self.fch2(inputs) #2
        inputs = self.bn1(inputs)
        inputs = self.fch2(inputs) #3
        inputs = self.bn1(inputs)
        inputs = self.fch2(inputs) #4
        inputs = self.bn1(inputs)
        inputs = self.fch2(inputs) #5
        inputs = self.bn1(inputs)
        inputs = self.fch2(inputs) #6
        inputs = self.bn1(inputs)
        inputs = self.fch2(inputs) #7
        inputs = self.bn1(inputs)
        inputs = self.fch2(inputs) #8
        
        outputs = self.fce(inputs)
        
        return outputs
  
torch.manual_seed(123)

pinn = FCN(2, 2, 20, 9)

pinn.eval()

"""Collocation Points"""
x_colloc = torch.linspace(0, 200, 500).view(-1,1).requires_grad_(True)
t_colloc = torch.linspace(0, 10, 500).view(-1,1).requires_grad_(True)

test_dataset = TrainData(x_train, t_train, h_train, q_train)
colloc_dataset = CollocData(x_colloc, t_colloc)

train_dataloader = DataLoader(test_dataset, batch_size = x_train.size(0))
colloc_dataloader = DataLoader(colloc_dataset, batch_size = x_colloc.size(0))

diam = 800*10**-3
Cs_A = np.pi*(diam**2)/4 #Cross sectional area
a = 1200 #wave speed in m/s
g = 9.81
f = 0.022
w_f = 1e-3

optimiser = torch.optim.Adam(pinn.parameters(), lr = 1e-3)

start_time = time.time()

pde_plot = []
data_plot = []

for i in range (500001):
    
    running_loss_pde = 0.0
    running_loss_data = 0.0
    
    for train_batch, colloc_batch in zip(train_dataloader, colloc_dataloader):
    
        optimiser.zero_grad()
        
        """PDE Loss"""        
        w_f = 1e-3
        w_f1 = 1e-4
        w_f2 = 1e-4
        
        train_x, train_t, target_h, target_q = train_batch
        colloc_x, colloc_t = colloc_batch
            
        colloc_output = pinn(colloc_x.unsqueeze(1), colloc_t.unsqueeze(1))
            
        h_hat, q_hat = colloc_output[:,0].unsqueeze(1), colloc_output[:,1].unsqueeze(1)
            
        dq_dt = torch.autograd.grad(q_hat, t_colloc, torch.ones_like(q_hat), create_graph=True)[0]
        dq_dx = torch.autograd.grad(q_hat, x_colloc, torch.ones_like(q_hat), create_graph=True)[0]
        dh_dt = torch.autograd.grad(h_hat, t_colloc, torch.ones_like(h_hat), create_graph=True)[0]
        dh_dx = torch.autograd.grad(h_hat, x_colloc, torch.ones_like(h_hat), create_graph=True)[0]
            
        F1 = (Cs_A * dq_dt) + (q_hat * dq_dx) + (g * Cs_A**2 * dh_dx) + (f * ((torch.abs(q_hat) * q_hat) / (2 * diam)))
        F2 = (Cs_A * dh_dt) + (q_hat * dh_dx) + ((a**2/g) * dq_dx)
            
        loss_pde = torch.mean(F1**2 + F2**2)
        
        """Data Loss"""
        train_output = pinn(train_x.unsqueeze(1), train_t.unsqueeze(1))
        hq_true = torch.cat([h_train, q_train], axis = 1) 
        loss_data = torch.mean((train_output - hq_true)**2)
        
        """Initial Condition"""
        init_hq = pinn(x_init, t_init)
        init_h, init_q = init_hq[:,0], init_hq[:,1]
        
        loss_init1 = ((torch.squeeze(init_h) - h_train[0]) + (torch.squeeze(init_q) - q_train[0]))**2
        
        dq_dt1 = torch.autograd.grad(init_q, t_init, torch.ones_like(init_q), create_graph = True)[0]
        dh_dt1 = torch.autograd.grad(init_h, t_init, torch.ones_like(init_q), create_graph = True)[0]
        
        loss_init2 = (torch.squeeze(dq_dt1) - 0)**2 + (torch.squeeze(dh_dt1) - 0)**2
        
        loss_ic = loss_init1 +loss_init2
        
        loss = loss_pde * w_f + loss_data + w_f1 * loss_init1 + w_f2 * loss_init2 
            
        loss.backward()
        optimiser.step()
        
        running_loss_pde += loss_pde.item() * x_colloc.size(0)
        running_loss_data += loss_data.item() * train_x.size(0)

        iteration_time = time.time() - start_time
        
    pde_plot.append(running_loss_pde / len(colloc_dataloader))
    data_plot.append(running_loss_data / len(train_dataloader))
    

    if i % 100 == 0:
        
        iteration_time = time.time() - start_time # Calculate time taken for the last 500 iterations
        print(f'Iteration [{i}], Data Loss: {loss_data.item()}, PDE Loss: {loss_pde.item()}, IC Loss: {loss_ic.item()}, Time to complete: {iteration_time:.4f}s'
                  )                
        start_time = time.time() # Reset the timer for the next block of 500 iterations
        
        indices_est = (train_x == 155).nonzero(as_tuple=False).squeeze()
        output_test = pinn(train_x.unsqueeze(1), train_t.unsqueeze(1)).detach()
        h_est, q_est = output_test[:,0][indices_est], output_test[:,1][indices_est]
        x_axis_est = train_t[indices_est]
            
        x_sorted, indices = torch.sort(x_axis_est)
        h_est = h_est[indices]
        q_est = q_est[indices]
        h_true = h_train[:len(x_sorted)].squeeze()
        q_true = q_train[:len(x_sorted)].squeeze()
            
            
        plt.figure(figsize=(6,2.5))        
        plt.plot(x_sorted, h_est, label="h_est")
        plt.plot(x_sorted, h_true, label="h_true", color = 'grey')
        plt.title(f"Pressure prediction - Training step {i}")
        plt.xlabel("Time (s)")
        plt.ylabel("Head (m)")
        plt.legend()
        plt.show()
        
        plt.figure(figsize=(6,2.5))
        plt.plot(x_sorted, q_true, label = "q_true", color = 'grey')
        plt.plot(x_sorted, q_est, label="q_est")
        plt.title(f"Flowrate prediction - Training step {i}")
        plt.xlabel("Time (s)")
        plt.ylabel("Flowrate (m3/s) (m)")
        plt.legend()
        plt.show()
          
    if i % 1000 == 0:
            
        epochs = list(range(len(pde_plot)))
        
        plt.figure(figsize=(6,2.5)) 
        plt.semilogy(pde_plot)
        plt.semilogy(data_plot)
        plt.plot(epochs, pde_plot, label="PDE Loss")
        plt.plot(epochs, data_plot, label="Data Loss")
        plt.title("Change in Loss over no. of Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
