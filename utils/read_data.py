import numpy as np
import torch
import torch.utils.data as Data
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

def get_data():

    # set seed
    seed = 2

    # set which file to use to build model and what is the grid size
    filenums = [1,2,3]
    gsize = [args.grid] #5,10,15,20
    overlapping_step = 3 # 1,3,5
    shuffle = True

    dataset_x = []
    dataset_y = []
    # MBP14 ： /Users/darren/资料/SPIF_DU/MainFolder/5mm_file/outfile3
    # /home/durrr/phd/SPIF_DU/MainFolder/50mm_file/outfile3/trainingfile_50mm_overlapping_5.txt
    for filenum in filenums:
         with open('/Users/darren/资料/SPIF_DU/MainFolder/{size}mm_file/outfile{fnum}/trainingfile_{size}mm_overlapping_{overlapping_step}.txt'.format(size = gsize, fnum = filenum,overlapping_step = overlapping_step), 'r') as f:
        # with open('/home/durrr/phd/SPIF_DU/MainFolder/{size}mm_file/outfile{fnum}/trainingfile_{size}mm_overlapping{overlapping_step}.txt'.format(size = gsize, fnum = filenum,overlapping_step = overlapping_step), 'r') as f:
            lines = f.readlines()
            if shuffle:
                random.Random(seed).shuffle(lines)
            else:
                pass
            # print(lines[10])
            for line in lines:
                line = line.strip("\n")
                dataset_x.append(line.split("|")[0].split(","))
                dataset_y.append(line.split("|")[1])
            


    # print(len(dataset_x))

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

