import numpy as np
import torch
import matplotlib as plt
import torch.nn.functional as F
import matplotlib.pyplot  as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import torch.utils.data as Data
from utils.read_data import get_data
from mamba_ssm import Mamba
from utils.cli import get_parser

parser = get_parser()
args = parser.parse_args()


gsize = [args.grid] #5,10,15,20

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
epo = 1000

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

model.eval()
pre,_ = model(x_test_tensor,None)
if torch.cuda.is_available():
    pre = pre.cpu()
pre = pre.detach().numpy()
y_test_tensor = y_test_tensor.cpu()

mae = mean_absolute_error(y_test_tensor,pre)
mse = mean_squared_error(y_test_tensor,pre)
rmse = mean_squared_error(y_test_tensor,pre,squared=False)
r2=r2_score(y_test_tensor,pre)

print("MAE",mae)
print("MSE",mse)
print("RMSE",rmse)
print("R2",r2)


# epos = np.arange(epo)+1
# mae_plt = plt.plot(epos,maes,label='MAE')
# mse_plt = plt.plot(epos,mses,label = 'MSE')
# rmse_plt = plt.plot(epos,rmses,label = 'RMSE')
# r2_plt = plt.plot(epos,r2s,label='R2')
# plt.title('Metrics_outfile{fnum}/gridized{size}mm'.format(size = gsize))
# plt.xlabel("Epo")
# plt.ylabel("Metrics Value")
# plt.legend()
# plt.show()
# plt.savefig("GRU_Metrics_outfile{filenum}_gridized{size}mm")