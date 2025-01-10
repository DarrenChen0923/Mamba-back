import numpy as np
import torch
import random
from utils.cli import get_parser

parser = get_parser()
args = parser.parse_args()

def create_dataset(X, y):
    features = []
    targets = []
    
    for i in range(0, len(X)): 
        data = [[i] for i in X[i]] # 序列数据  
        label = [y[i]] # 标签数据
        
        # 保存到features和labels
        features.append(data)
        targets.append(label)
    
    return np.array(features,dtype=np.float32), np.array(targets,dtype=np.float32)


# split data
# x_train, x_test, y_train, y_test

def split_dataset(x, y, train_ratio=0.8):

    x_len = len(x) # 特征数据集X的样本数量
    train_data_len = int(x_len * train_ratio) # 训练集的样本数量
    
    x_train = x[:train_data_len] # 训练集
    y_train = y[:train_data_len] # 训练标签集
    
    x_test = x[train_data_len:] # 测试集
    y_test = y[train_data_len:] # 测试集标签集
    
    # 返回值
    return x_train, x_test, y_train, y_test

def normalize_data(data, overall_max, overall_min):
    # 对形状为 (64, 9, 1) 的数据进行归一化
    normalized_data = (data - overall_min) / (overall_max - overall_min)
    return normalized_data

def get_data():

    # set seed
    seed = 2

    # set which file to use to build model and what is the grid size
    filenums = [1,2,3]
    gsize = args.grid #5,10,15,20
    shuffle = True

    dataset_x = []
    dataset_y = []

    for filenum in filenums:
         temp_x = []
         with open(args.project_root+'/Mamba-back/data/{size}mm_file/outfile{fnum}/trainingfile_{size}mm_overlapping_3.txt'.format(size = gsize, fnum = filenum), 'r') as f:
            lines = f.readlines()
            if shuffle:
                random.Random(seed).shuffle(lines)
            else:
                pass
            for line in lines:
                line = line.strip("\n")
                x = line.split("|")[0].split(",")
                y = line.split("|")[1]

                # 检查 x 中是否包含 'NaN'，并将其替换为 0
                x = [0 if value == 'NaN' else value for value in x]
                
                # 如果 y 是 'NaN'，则将其赋值为 0
                y = 0 if y == 'NaN' else y
                temp_x.append(x)
                dataset_x.append(x)
                dataset_y.append(y)

            

         print(len(temp_x))
    print(len(dataset_y))

    lable = [float(y) for y in dataset_y]
    input_x = []
    for grp in dataset_x:
        input_x.append([float(z) for z in grp])


    input_x,lable = create_dataset(input_x, lable)
    x_train, x_test, y_train, y_test = split_dataset(input_x, lable, train_ratio=0.80)

    nsample,nx,ny = x_train.shape
    x_train_2d = x_train.reshape(nsample, nx*ny)

    nsamplet,nxt,nyt = x_test.shape
    x_test_2d = x_test.reshape(nsamplet, nxt*nyt)


    #tensor to numpy
    x_train_tensor = torch.from_numpy(x_train_2d)
    x_test_tensor = torch.from_numpy(x_test_2d)
    y_train_tensor = torch.from_numpy(y_train)
    y_test_tensor = torch.from_numpy(y_test)

    #gpu environment: transfer int cuda
    if torch.cuda.is_available():
        x_train_tensor = x_train_tensor.cuda()
        x_test_tensor = x_test_tensor.cuda()
        y_train_tensor = y_train_tensor.cuda()
        y_test_tensor = y_test_tensor.cuda()
    
    
    return x_train_tensor,y_train_tensor,x_test_tensor,y_test_tensor


def find_min_max_from_data(data_tensor):
    max_values = []
    min_values = []
    
    for line in data_tensor:
        numbers = line[:9]
        
        # 计算最大最小值
        max_values.append(torch.max(numbers))
        min_values.append(torch.min(numbers))
    
    # 最终最大值和最小值
    overall_max = torch.max(torch.stack(max_values))
    overall_min = torch.min(torch.stack(min_values))
    
    return overall_max.item(), overall_min.item()

