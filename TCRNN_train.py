import torch
import os
import torch.utils.data as Data
from TCRNN_model import TCRNN2, Loss_TCRNN2
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn import preprocessing
plt.rcParams.update({"font.size": 24,
                     "font.family": "sans-serif"}) # fontsize for figures
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)


#%%============================================================================
### Parameters
loadCheckPoint = 0     # 1: Load saved NN model
use_gpu = 0            # 1: Use gpu for computation
RNN_STEPS = 10         # including current step
BATCH_SIZE = 128       # training batch size
N_EPHOCS = 50          # number of training epochs
LR = 1e-3              # initial learning rate
wd = 1e-5              # weight decay for regularization
flag_plot_sample = 0   # whether to plot data
lamb = [1,1,1]         # regularization parameters in the loss function
ns = 5                 # number of noisy stress data
rs = 0.3               # random level of Gaussian noise for shear stress
rv = 0.3               # random level of Gaussian noise for vertical stress
plot_noisy_data = False  # plot noisy data

### Model architecture
# RNN (or GRU) for inference of internal state (z)
# se: shear strain
# ve: vertical strain
# ss: shear stress
# vs: vertical stress
rnn_input = 4       # input size of history steps: (se, ve, ss, vs)
rnn_final_input = 2 # input size of final step: (se, ve)
rnn_output = 2      # output size: (z)
rnn_layer = 1       # number of rnn stacked together
rnn_hidden = 30     # hidden state dimension
rnn_units = [rnn_input,rnn_hidden,rnn_output] # number of units in input, hidden, and output layers
rnn_str = str(rnn_units).replace(', ','-')
rnn_final_unit = [rnn_units[1],rnn_hidden,rnn_output] # GRU

# NN for prediction of free energy (F)
nn_input = rnn_final_input + rnn_output # (se, ve, z)
nn_output = 1 # F
nn_hidden = rnn_hidden
nn_units = [nn_input,nn_hidden,nn_output] 
nn_str = str(nn_units).replace(', ','-')

# for generating dataset
RNN_INPUTS = rnn_input # (se, ve, ss, vs)
RNN_OUTPUTS = 2 # output dimension of the whole model: (ss, vs)

params = {}
params['RNN_STEPS'] = RNN_STEPS
params['BATCH_SIZE'] = BATCH_SIZE
params['N_EPHOCS'] = N_EPHOCS
params['LR'] = LR
params['wd'] = wd
params['lamb'] = lamb
params['rnn_final_input'] = rnn_final_input
params['rnn_units'] = rnn_units
params['rnn_str'] = rnn_str
params['RNN_INPUTS'] = RNN_INPUTS
params['RNN_OUTPUTS'] = RNN_OUTPUTS
params['nn_units'] = nn_units

#============================================================================
### GPU Setting
if use_gpu == 1:
    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()
    
    # select gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.cuda.init() # initialize gpu
    
#%%============================================================================
### Data loading and preprocessing
trainID = [15,17]
testID = [16]

### Read training data
# [time, CSR, shear strain (se), vert strain (ve), shear stress (ss), vert stress (vs)]
train0 = np.empty((0,6),dtype='float32')
train0b = []
np_train = []
for i in trainID:
    train_filename = f'eo_0_601_sigv_40_CSR_0_{i}'
    train_csr = 'CSR: 0.' + train_filename.split('_')[-1]
    data = pd.read_csv(f'./data/{train_filename}.csv')
    data.dropna(axis=1, inplace=True)
    data.insert(loc=0, column='CSR', value=i*np.ones((data.shape[0])))
    tmp = data.iloc[:,[0,1,2,4,3,5]].to_numpy(dtype='float32')
    
    train0 = np.append(train0, tmp, axis=0)
    train0b.append(tmp)
    np_train.append(data.shape[0])
    
np_train = np.asarray(np_train)      # number of samples in training cases
np_train2 = np_train - (RNN_STEPS-1) # number of training samples in training cases

### Read testing data
test0 = np.empty((0,6),dtype='float32')
test0b = []
np_test = []
for i in testID:
    test_filename = f'eo_0_601_sigv_40_CSR_0_{i}'
    test_csr = 'CSR: 0.' + test_filename.split('_')[-1]
    data = pd.read_csv(f'./data/{test_filename}.csv')
    data.dropna(axis=1, inplace=True)
    data.insert(loc=0, column='CSR', value=i*np.ones((data.shape[0])))
    tmp = data.iloc[:,[0,1,2,4,3,5]].to_numpy(dtype='float32')
    
    test0 = np.append(test0, tmp, axis=0)
    test0b.append(tmp)
    np_test.append(data.shape[0])

np_test = np.asarray(np_test)      # number of samples in testing cases
np_test2 = np_test - (RNN_STEPS-1) # number of testing samples in testing cases

#==============================================================================
### Standardization
print('Before Standardization *************************************')
with np.printoptions(formatter={'float': '{: 0.3f}'.format}, suppress=True):
    print('train0 mean: ', train0.mean(axis=0))
    print('train0 std: ', train0.std(axis=0))
    print('test0 mean: ', test0.mean(axis=0))
    print('test0 std: ', test0.std(axis=0))
print()

train = np.copy(train0)
test = np.copy(test0)
train_scaler = preprocessing.StandardScaler().fit(train0[:,2:])
train[:,2:] = train_scaler.transform(train0[:,2:])
test[:,2:] = train_scaler.transform(test0[:,2:])

trainb = []
testb = []
for i in range(len(trainID)):
    tmp = np.copy(train0b[i])
    tmp[:,2:] = train_scaler.transform(tmp[:,2:])
    trainb.append(tmp)
for i in range(len(testID)):
    tmp = np.copy(test0b[i])
    tmp[:,2:] = train_scaler.transform(tmp[:,2:])
    testb.append(tmp)
    
print('After Standardization **************************************')
with np.printoptions(formatter={'float': '{: 0.3f}'.format}, suppress=True):
    print('train mean: ', train.mean(axis=0))
    print('train std: ', train.std(axis=0))
    print('test mean: ', test.mean(axis=0))
    print('test std: ', test.std(axis=0))
print()

### Scaler Statistics
std_e = train_scaler.scale_[0:2]    # std for strain
std_s = train_scaler.scale_[2:4]    # std for stress
mean_e = train_scaler.mean_[0:2]    # mean for strain
mean_s = train_scaler.mean_[2:4]    # mean for stress

#==============================================================================
# Add Gaussian noise to stresses
def add_noise(d0):
    d = np.copy(d0)
    stress_std = rs*np.sqrt(np.abs(d[:,3]).max()) # shear stress
    d[1:,3] = d[1:,3] + np.random.normal(scale=stress_std,
                                           size=d.shape[0]-1)
    
    stress_std = rv*np.sqrt(np.abs(d[:,4]).max()) # vertical stress
    d[1:,4] = d[1:,4] + np.random.normal(scale=stress_std,
                                           size=d.shape[0]-1)
    
    d[:,1:] = train_scaler.transform(d[:,1:])
    return d

### plot noisy data
if plot_noisy_data:
    idx = np.where(train0[:,0]==trainID[0])[0]
    t = train0[idx,1]
    data_noisy = add_noise(train0[idx,1:]) # excluding CSR column
    data0_noisy = np.copy(data_noisy)
    data0_noisy[:,1:] = train_scaler.inverse_transform(data_noisy[:,1:])
    fig = plt.figure()
    plt.plot(t,train0[idx,4],'k-',label='Data',alpha=0.6)
    plt.plot(t,data0_noisy[:,3],'r.',label='Perturbation',alpha=0.4)
    plt.xlabel('time',fontsize=18)
    plt.ylabel('shear stress',fontsize=18)
    
    fig = plt.figure()
    plt.plot(t,train0[idx,5],'k-',label='Data',alpha=0.6)
    plt.plot(t,data0_noisy[:,4],'r.',label='Perturbation',alpha=0.4)
    plt.xlabel('time',fontsize=18)
    plt.ylabel('vertical stress',fontsize=18)
    
    fig = plt.figure()
    plt.plot(train0[idx,2],train0[idx,4],'k-',label='Data',alpha=0.6)
    plt.plot(data0_noisy[:,1],data0_noisy[:,3],'r.',label='Perturbation',alpha=0.4)
    plt.xlabel('shear strain',fontsize=18)
    plt.ylabel('shear stress',fontsize=18)
    plt.legend(fontsize=18)

### Set input and output af training and testing datasets
def create_dataset(dataset, N_STEPS, N_INPUTS, N_OUTPUTS, look_back=1):
    dataX = np.empty((0,N_STEPS,N_INPUTS),dtype='float32')
    dataY = np.empty((0,N_OUTPUTS),dtype='float32')
    
    for i in range(len(dataset)-look_back+1):
        # Input: (shear strain, vertical strain, shear stress, vertical stress)
        a = dataset[i:(i+look_back), [1,2,3,4]].reshape(1,N_STEPS,N_INPUTS)
        dataX = np.append(dataX, a, axis=0)
        
        # Output: (shear stress,vertical stress)
        b = dataset[(i+look_back-1) : (i+look_back), [3,4]].reshape((-1,N_OUTPUTS))
        dataY = np.append(dataY, b, axis=0)
    return dataX, dataY

look_back = RNN_STEPS
trainX = np.empty((0,RNN_STEPS,RNN_INPUTS),dtype='float32')
trainY = np.empty((0,RNN_OUTPUTS),dtype='float32')
trainX2 = []
trainY2 = []
trainX0 = np.empty((0,RNN_STEPS,RNN_INPUTS),dtype='float32')
trainY0 = np.empty((0,RNN_OUTPUTS),dtype='float32')
for j,i in enumerate(trainID):
    data_t0 = train0[train0[:,0]==i,1:] # excluding CSR column
    data_t = train[train[:,0]==i,1:]    # excluding CSR column
    tempX, tempY = create_dataset(data_t, RNN_STEPS, RNN_INPUTS, 
                                  RNN_OUTPUTS, look_back)
    tempX0, tempY0 = create_dataset(data_t0, RNN_STEPS, RNN_INPUTS, 
                                    RNN_OUTPUTS, look_back)
    trainX = np.append(trainX,tempX,axis=0)
    trainY = np.append(trainY,tempY,axis=0)
    
    trainX2.append(tempX)
    trainY2.append(tempY)
    trainX0 = np.append(trainX0,tempX0,axis=0) # original scale
    trainY0 = np.append(trainY0,tempY0,axis=0) # original scale
    
    for j in range(ns):
        train_noisy = add_noise(np.copy(data_t0))
        tempXs,_ = create_dataset(train_noisy, RNN_STEPS, RNN_INPUTS, 
                                  RNN_OUTPUTS, look_back)
        trainX = np.append(trainX,tempXs,axis=0)
        trainY = np.append(trainY,tempY,axis=0)

testX = np.empty((0,RNN_STEPS,RNN_INPUTS),dtype='float32')
testY = np.empty((0,RNN_OUTPUTS),dtype='float32')
testX2 = []
testY2 = []
testX0 = np.empty((0,RNN_STEPS,RNN_INPUTS),dtype='float32')
testY0 = np.empty((0,RNN_OUTPUTS),dtype='float32')
for j,i in enumerate(testID):
    tempX, tempY = create_dataset(test[test[:,0]==i,1:], RNN_STEPS, 
                                  RNN_INPUTS, RNN_OUTPUTS, look_back)
    tempX0, tempY0 = create_dataset(test0[test0[:,0]==i,1:], RNN_STEPS, 
                                    RNN_INPUTS, RNN_OUTPUTS, look_back)
    testX = np.append(testX,tempX,axis=0)
    testY = np.append(testY,tempY,axis=0)
    testX2.append(tempX)
    testY2.append(tempY)
    testX0 = np.append(testX0,tempX0,axis=0) # original scale
    testY0 = np.append(testY0,tempY0,axis=0) # original scale

#==============================================================================
### Convert data to tensor form
if use_gpu == 1:
    trainX = torch.from_numpy(trainX).cuda()
    trainY = torch.from_numpy(trainY).cuda()
    testX = torch.from_numpy(testX).cuda()
    testY = torch.from_numpy(testY).cuda()
else:
    trainX = torch.from_numpy(trainX).float()
    trainY = torch.from_numpy(trainY).float()
    testX = torch.from_numpy(testX).float()
    testY = torch.from_numpy(testY).float()
    
torch_dataset = Data.TensorDataset(trainX, trainY)
loader = Data.DataLoader(
    dataset=torch_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, num_workers=0)

#%%============================================================================
### RNN Model instance, Activation: 
# act_type = 1 (ReLU), 2 (Tanh), 3 (Sigmoid), 4 (ELU), 5 (SiLU), 6 (ELU2), 
# 7 (SiLU2), 8 (ELU3)
model = TCRNN2(use_gpu, train_scaler, nn_units, rnn_units, rnn_final_input, 
               rnn_final_unit, rnn_layer, act_type=5)
loss_func = Loss_TCRNN2(use_gpu)
if use_gpu == 1:
    model = model.cuda()
    loss_func = loss_func.cuda()

print(model)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=wd)

# learning rate scheduler: 
# Scheduler 2: Reduce learning rate when a metric has stopped improving
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                 factor=0.1,
                                                 patience=10, 
                                                 threshold=1e-4)

#==============================================================================
# Load Saved Model if Specified
if loadCheckPoint == 1: # load a saved NN model
    optimizer = torch.optim.Adam(model.parameters())
    checkpoint = torch.load('checkpoint.pth.tar')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    lossTrain = checkpoint['lossTrain']
    lossTest = checkpoint['lossTest']
    lr_hist = checkpoint['lr_hist']
    train_scaler = checkpoint['train_scaler']
    lamb = checkpoint['lamb']
    try:
        lossTrain = lossTrain.tolist()
        lossTest = lossTest.tolist()
        lr_hist = lr_hist.squeeze().tolist()
    except: 
        pass
else:
    lossTrain = []
    lossTest = []
    lr_hist = []
    
#%%============================================================================
### Training
startT = time.time()
model.train()
for epoch in range(N_EPHOCS):  # loop over the dataset multiple times
    for step, (b_x, b_y) in enumerate(loader): # for each training step
        optimizer.zero_grad() # zero the parameter gradients
        inputs = b_x.view(-1,RNN_STEPS,RNN_INPUTS)
        outputs = model(inputs)     # input x and predict based on x
        loss = loss_func(lamb,b_y,outputs)     # must be (1. nn output, 2. target)
        loss.backward(retain_graph=True)         # backpropagation, compute gradients
        optimizer.step()        # apply gradients
        
    lossTrain.append(loss_func(lamb,trainY,model(trainX)).data.numpy())
    lossTest.append(loss_func(lamb,testY,model(testX)).data.numpy())
    
    scheduler.step(lossTest[-1]) # update learning rate by scheduler 2
    lr_hist.append(scheduler.optimizer.param_groups[0]['lr'])
    
    print(f"Epoch: {len(lossTrain)} | LR: {scheduler.optimizer.param_groups[0]['lr']:.1e} | Scaled Train Loss: {lossTrain[-1]:.4f}")
    print(f"Epoch: {len(lossTest)} | LR: {scheduler.optimizer.param_groups[0]['lr']:.1e} | Scaled Test Loss: {lossTest[-1]:.4f}")
    
endT = time.time()
print(f'\n********* Training time: {endT-startT} s,  {(endT-startT)/60} mins, {(endT-startT)/3600} hrs')

#==============================================================================
# Save Check Point of model, which can be loaded for continuous training
torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lossTrain': lossTrain,
            'lossTest':lossTest,
            'lr_hist':lr_hist,
            'train_scaler':train_scaler,
            'lamb':lamb,
            'testID':testID,
            'trainID':trainID,
            'params':params,
            }, 'checkpoint.pth.tar')
    
#%%============================================================================
### Plot history of training loss
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
ax.plot(range(len(lossTrain)), lossTrain, 'b-', lw=3, label = 'Train')
ax.plot(range(len(lossTrain)), lossTest, 'r-', lw=3, label = 'Test')
ax.set_xlabel('Epochs'); ax.set_ylabel('Loss'); plt.grid()
ax.set_xlim(0,len(lossTrain))
ax.set_yscale('log')
plt.legend()
#plt.savefig(f'./fig/loss_{rnn_str}_nstep{RNN_STEPS}_ep{N_EPHOCS}_lr{LR}.png',bbox_inches='tight')
plt.show()

### Plot history of learning rate
plt.rcParams.update({'font.size': 20})
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
ax.plot(range(len(lossTrain)), lr_hist, 'k-', lw=3)
ax.set_xlabel('Epochs'); ax.set_ylabel('Learning rate'); plt.grid()
ax.set_xlim(0,len(lossTrain))
ax.set_yscale('log')
plt.legend(frameon=False)
#plt.savefig(f'./fig/lr_hist_{rnn_str}_nstep{RNN_STEPS}_ep{N_EPHOCS}_lr{LR}.png',bbox_inches='tight')
plt.show()