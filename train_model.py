import os
import torch
import torch.nn as nn
from time  import time
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import torch.utils.data as Data
from utils.read_data import get_data
from utils.read_data import normalize_data
from utils.read_data import find_min_max_from_data
from models.mamba_back_model import MambaModel
from torch.utils.tensorboard import SummaryWriter
from utils.cli import get_parser

parser = get_parser()
args = parser.parse_args()

gsize = args.grid
d_model = args.d_model
#Get data
x_train_tensor,y_train_tensor,x_test_tensor,y_test_tensor = get_data() #shape: (size,1)

train_dataset = Data.TensorDataset(x_train_tensor,y_train_tensor)
test_dataset = Data.TensorDataset(x_test_tensor,y_test_tensor)


# set paramemaers
batch = 64

writer = SummaryWriter('runs/mamba_model_experiment')

# optimizer
lr = 0.00001

#complie
epo = 1500

#Create dataloader
loader = Data.DataLoader(dataset=train_dataset,batch_size=batch,shuffle=True)

#Model#
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MambaModel(
    # This module uses roughly 3 * expand * d_model^2 parameters
    d_model=d_model, # Model dimension d_model
    d_state=16,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
).to(device=device)

# print(model)


criterion = nn.L1Loss()

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
time0 = time()
overall_max, overall_min = find_min_max_from_data(x_train_tensor)

update_step = 0
x_test_tensor = x_test_tensor.unsqueeze(-1)
overall_max, overall_min = find_min_max_from_data(x_test_tensor)
x_test_tensor = normalize_data(x_test_tensor, overall_max, overall_min)
y_test_tensor = y_test_tensor.cpu().detach().numpy()

with tqdm(range(epo), unit="epoch") as tepochs:
    for e in tepochs:
        total_loss = 0
        with tqdm(loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {e+1}")
            for step, (batch_x, batch_y) in enumerate(tepoch):
                batch_x = batch_x.unsqueeze(-1)

                batch_x = normalize_data(batch_x,overall_max, overall_min)
                optimizer.zero_grad()
                
                output = model(batch_x)
                loss = criterion(output, batch_y)
                total_loss += loss
                
                writer.add_scalar('loss/train', loss.item(), update_step)
                update_step += 1
                loss.backward()
                optimizer.step()
                
                tepoch.set_postfix(loss=loss.item())
        #evaluation 
        model.eval()
        pre = model(x_test_tensor)
        pre = pre.cpu().detach().numpy()
        mse = mean_squared_error(y_test_tensor,pre)
        writer.add_scalar('loss/eval', mse.item(), update_step)
        model.train()
    
# saving the model
os.makedirs("trained_models",exist_ok=True)
torch.save(model.state_dict(), f'trained_models/mamba_back_{gsize}mm_overlapping_3_dModel_{d_model}.pth')

print(f"Model saved in trained_models/mamba_back_{gsize}mm_overlapping_3_dModel_{d_model}.pth!")

model.eval()
x_train_tensor = x_train_tensor.unsqueeze(-1)

x_train_tensor = normalize_data(x_train_tensor, overall_max, overall_min)
pre = model(x_train_tensor)
pre = pre.cpu().detach().numpy()
y_train_tensor = y_train_tensor.cpu().detach().numpy()

mae = mean_absolute_error(y_train_tensor,pre)
mse = mean_squared_error(y_train_tensor,pre)
rmse = mean_squared_error(y_train_tensor,pre,squared=False)
r2=r2_score(y_train_tensor,pre)

print("Training Performance:")
print("MAE",mae)
print("MSE",mse)
print("RMSE",rmse)
print("R2",r2)
