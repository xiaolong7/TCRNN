import torch
import torch.nn as nn
import numpy as np
from torch.autograd import grad
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

#%%============================================================================
# activation functions
class ELU2(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        f = nn.ELU()
        y = f(x)**2
        return y
    
class ELU3(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        y = torch.exp(x) - 1
        y[x>=0] = x[x>=0]**2
        return y
    
class SiLU2(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        f = nn.SiLU()
        y = f(x)**2
        return y

#%% Multi-layer Feedforward Networks
class Net(nn.Module):
    def __init__(self, units, act_type=1):
        super(Net,self).__init__()
        self.act_type = act_type
        if self.act_type == 1:
            self.act = nn.ReLU()
        elif self.act_type == 2:
            self.act = nn.Tanh()
        elif self.act_type == 3:
            self.act = nn.Sigmoid()
        elif self.act_type == 4:
            self.act = nn.ELU()
        elif self.act_type == 5:
            self.act = nn.SiLU()
        elif self.act_type == 6:
            self.act = ELU2()
        elif self.act_type == 7:
            self.act = SiLU2()
        elif self.act_type == 8:
            self.act = ELU3() 
        else:
            self.act = nn.Linear() 
        self.hidden = self._make_layers(units)
        self.fc = nn.Linear(units[-2],units[-1])
        
    def _make_layers(self, units):
        layers = []
        for i in range(len(units)-2):
            layers.append(nn.Linear(units[i],units[i+1]))
            layers.append(self.act)
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.hidden(x)
        x = self.fc(x)
        return x
    
#%%============================================================================
# GRU: predicts internal states at the last two steps to compute dz=z_new-z_old
class GRU(nn.Module):
    def __init__(self, gpu, train_scaler, gru_unit, gru_layers, 
                 final_input_size, final_unit, act_type=2):
        super().__init__()
        self.gpu = gpu
        self.n_layers = gru_layers
        self.n_inputs = gru_unit[0]
        self.n_hidden = gru_unit[1]
        self.n_outputs = gru_unit[-1]
        self.final_input_size = final_input_size
        self.act_type = act_type
        if self.act_type == 1:
            self.act = nn.ReLU()
        elif self.act_type == 2:
            self.act = nn.Tanh()
        elif self.act_type == 3:
            self.act = nn.Sigmoid()
        elif self.act_type == 4:
            self.act = nn.ELU()
        elif self.act_type == 5:
            self.act = nn.SiLU()
        elif self.act_type == 6:
            self.act = ELU2()
        elif self.act_type == 7:
            self.act = SiLU2()   
        elif self.act_type == 8:
            self.act = ELU3()   
        self.gru = nn.GRU(self.n_inputs, self.n_hidden, self.n_layers, 
                          batch_first=True, dropout=0.0) # defualt: num_layers=1, bias=True
        self.Wx = nn.Linear(final_input_size, self.n_hidden)
        self.Wh = nn.Linear(self.n_hidden, self.n_hidden)
        self.Wz = Net(final_unit, act_type=act_type)
        
    def _make_layers(self, units):
        layers = []
        for i in range(len(units)-1):
            layers.append(nn.Linear(units[i],units[i+1]))
            layers.append(self.act)
        layers.pop() # linear activation for the last layer
        return nn.Sequential(*layers)
    
    def forward(self, x):       
        # x shape: [batch_size, n_steps, n_inputs]
        hidden = torch.zeros(self.n_layers, x.size(0), self.n_hidden)
        if self.gpu:
            hidden = hidden.cuda()
            
        # history steps
        _,hn = self.gru(x[:,:-1,:self.n_inputs], hidden) # (e,s)
        h_old = hn[-1].view(-1, self.n_hidden)
        z_old = self.Wz(h_old).view(-1, self.n_outputs) # internal state at previous step
        
        # final step
        inputs = x[:,-1,:self.final_input_size] # (e)
        h_new = self.Wx(inputs) + self.Wh(h_old)
        h_new = self.act(h_new)
        z_new = self.Wz(h_new).view(-1, self.n_outputs)
        
        # dzdt
        dzdt_new = z_new - z_old
        
        return z_new, dzdt_new # (batch_size, n_output) 

#%%===========================================================================
# Thermodynamically consistent RNN (model 2): uses GRU with dz calculated
class TCRNN2(nn.Module): 
    def __init__(self, gpu, train_scaler, nn_units, rnn_units, rnn_final_input, 
                 rnn_final_unit, rnn_layer, act_type=2):
        super().__init__()
        self.train_scaler = train_scaler
        self.gpu = gpu
        self.nn_units = nn_units
        self.rnn_units = rnn_units
        self.rnn_final_input = rnn_final_input
        self.rnn_model = GRU(gpu, train_scaler, rnn_units, rnn_layer,
                              rnn_final_input, rnn_final_unit, 
                              act_type=act_type)
        self.nn_model = Net(nn_units, act_type=act_type)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        train_scaler_mean = torch.from_numpy(self.train_scaler.mean_).float()
        train_scaler_scale = torch.from_numpy(self.train_scaler.scale_).float()
        if self.gpu:
            train_scaler_mean = train_scaler_mean.cuda()
            train_scaler_scale = train_scaler_scale.cuda()
        std_e = train_scaler_scale[0:2][None,:]    # std for strain
        std_s = train_scaler_scale[2:4][None,:]    # std for stress
        mean_s = train_scaler_mean[2:4][None,:]    # mean for stress
        
        ### current internal state variable and its rates from RNN
        z_hat_new,dzdt_new = self.rnn_model(x)
        
        ### Energy potential from NN
        e_new = x[:,-1,:2] # (batch_size, 2)
        inputs = torch.cat((e_new,z_hat_new), dim=1) # (batch_size, 3)
        inputs.requires_grad_(True)
        F_hat0 = self.nn_model(inputs)
        
        ### Gradient of energy potential
        external_grad = torch.ones_like(F_hat0)
        if self.gpu:
            external_grad = external_grad.cuda()
        df = grad(F_hat0, inputs, grad_outputs=external_grad, 
                      retain_graph=True, create_graph=True, 
                      allow_unused=True)[0]
        
        ### Stress
        dfde = df[:,:2]
        s_hat0 = dfde / std_e
        s_hat = (s_hat0 - mean_s) / std_s
        
        ### Dissipation rate
        dfdz = df[:,2:2+self.rnn_units[-1]]
        D_hat0 = - torch.matmul(dfdz[:,None,:], dzdt_new[:,:,None]).squeeze(dim=-1)
        
        return s_hat,z_hat_new,F_hat0,D_hat0,dzdt_new

#%% Thermodynamically consistent loss functions
### loss function for TCRNN2 (dz)
class Loss_TCRNN2(nn.Module):
    def __init__(self, gpu):
        super().__init__()
        self.gpu = gpu
        # self.loss = nn.MSELoss(reduction="mean")
        self.loss = nn.L1Loss(reduction="mean")
        self.relu = nn.ReLU()
        
    def forward(self, lamb, target, outputs):
        s = target.squeeze()
        s_hat = outputs[0].squeeze()
        F_hat0 = outputs[2].squeeze()
        D_hat0 = outputs[3].squeeze()
        
        loss1 = self.loss(s_hat, s)
        loss2 = self.relu(-F_hat0).mean()
        loss3 = self.relu(-D_hat0).mean()
        
        tot_loss = lamb[0]*loss1 + lamb[1]*loss2 + lamb[2]*loss3
        return tot_loss