import torch
import torch.nn.functional as F
from models.mamba_back_model import MambaModel
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import torch.utils.data as Data
from utils.read_data import get_data
from utils.read_data import normalize_data
from utils.read_data import find_min_max_from_data
from mamba_ssm import Mamba
from utils.cli import get_parser

parser = get_parser()
args = parser.parse_args()

# /home/duchen /Mamba-back/trained_models/
model_path = args.project_root + f'/Mamba-back/trained_models/{args.load_model}'
gsize = args.grid #5,10,15,20
d_model = args.d_model
#Get data
x_train_tensor,y_train_tensor,x_test_tensor,y_test_tensor = get_data()

train_dataset = Data.TensorDataset(x_train_tensor,y_train_tensor)
test_dataset = Data.TensorDataset(x_test_tensor,y_test_tensor)


#Model#
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MambaModel(
    # This module uses roughly 3 * expand * d_model^2 parameters
    d_model=d_model, # Model dimension d_model
    d_state=16,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
).to(device=device)
model.load_state_dict(torch.load(model_path))
model.eval()
x_test_tensor = x_test_tensor.unsqueeze(-1)
overall_max, overall_min = find_min_max_from_data(x_test_tensor)
x_test_tensor = normalize_data(x_test_tensor, overall_max, overall_min)
pre = model(x_test_tensor)
pre = pre.cpu().detach().numpy()
y_test_tensor = y_test_tensor.cpu().detach().numpy()

mae = mean_absolute_error(y_test_tensor,pre)
mse = mean_squared_error(y_test_tensor,pre)
rmse = mean_squared_error(y_test_tensor,pre,squared=False)
r2=r2_score(y_test_tensor,pre)

print("Test Performance:")
print("MAE",mae)
print("MSE",mse)
print("RMSE",rmse)
print("R2",r2)