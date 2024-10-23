import torch
import torch.nn as nn
from torch import optim
from time  import time
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import torch.utils.data as Data
from utils.read_data import get_data
from mamba_ssm import Mamba

#Get data
x_train_tensor,y_train_tensor,x_test_tensor,y_test_tensor = get_data()

train_dataset = Data.TensorDataset(x_train_tensor,y_train_tensor)
test_dataset = Data.TensorDataset(x_test_tensor,y_test_tensor)

# set paramemaers
gru_units = 128
num_layer = 1
acti = 'relu'
drop = 0
regu = None #regularizers.l2(1e-4)
batch = 8
#16！

# optimizer
lr = 0.001
mom = 0.01

#complie
callback_file = None
epo = 1000

#Create dataloader
loader = Data.DataLoader(dataset=train_dataset,batch_size=batch,shuffle=True)

#Model#
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch, length, dim = 2, 64, 16
x = torch.randn(batch, length, dim).to("cuda")
model = Mamba(
    # This module uses roughly 3 * expand * d_model^2 parameters
    d_model=dim, # Model dimension d_model
    d_state=16,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
).to(device=device)

y = model(x)
assert y.shape == x.shape

print(model)


criterion = nn.L1Loss()

optimize = optim.SGD(model.parameters(),lr =lr,momentum=mom,nesterov=False)
time0 = time()

hx  = x_train_tensor
cx  = torch.randn((x_train_tensor.shape[0],x_train_tensor.shape[1]))

maes = []
mses = []
rmses = []
r2s = []

#metrics
def metrics(predict,expected):
    if torch.cuda.is_available():
        predict = predict.cpu()
    predict = predict.detach().numpy()
    expected = expected.cpu()

    mae = mean_absolute_error(expected,predict)
    mse = mean_squared_error(expected,predict)
    rmse = mean_squared_error(expected,predict,squared=False)
    r2=r2_score(expected,predict)

    maes.append(mae)
    mses.append(mse)
    rmses.append(rmse)
    r2s.append(r2)
   
# total_losss=[]

for e in range(epo):
   total_loss = 0
   for step, (batch_x,batch_y) in enumerate(loader):
    optimize.zero_grad()
    
    hx,cx = model(batch_x)
    
    metrics(hx,batch_y)
    loss= criterion(hx,batch_y)
    loss.backward()

    optimize.step()
    total_loss = total_loss + loss.item()
    # if step == len(loader) -1:
    #    total_losss.append(total_loss)
    print(f"Epoch: {e+1}, Step: {step}, Loss:{loss.item()}")

print("\nTraining Time(in minutes) = ",(time()-time0)/60)


