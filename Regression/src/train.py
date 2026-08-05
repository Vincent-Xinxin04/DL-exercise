import math
import numpy as np

import pandas as pd
import os

#进度条
from tqdm import tqdm

#引用pytorch的一些库
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

#拟合训练曲线
from torch.utils.tensorboard import SummaryWriter 


#设置随机种子的函数，可以复现项目
def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

#将训练数据划分为训练集和验证集
def train_valid_split(data_set,valid_ratio,seed):
    valid_set_size = int(valid_ratio * len(data_set))
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set, [train_set_size,valid_set_size],generator=torch.Generator().manual_seed(seed))
    return data_set[train_set.indices], data_set[valid_set.indices]


#数据集准备
class Covid19Dataset(Dataset):
    def __init__(self,x,y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)
    def __getitem__(self,idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx],self.y[idx]
    def __len__(self):
        return len(self.x)

#定义模型
class mymodel(nn.Module):
    def __init__(self,input_dim):
        super(mymodel,self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim,32),
            nn.ReLU(),
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,1)
        )
    def forward(self,x):
        x = self.layers(x)
        x = x.squeeze(1)
        return x

#筛选特征
def select_feature(train_data,valid_data,test_data,select_all = True):
    y_train , y_valid = train_data[:,-1] , valid_data[:,-1]
    raw_x_train , raw_x_valid , raw_x_test = train_data[:,:-1] , valid_data[:,:-1] , test_data
    if select_all:
        feature_idx = list(range(raw_x_train.shape[1]))
    else:
        feature_idx = [] #这里自定义哪几个特征好

    return raw_x_train[:,feature_idx], raw_x_valid[:,feature_idx], raw_x_test[:,feature_idx], y_train, y_valid

#训练循环
def train(train_loder,valid_loder,model,config,device):
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
    writer = SummaryWriter()
    if not os.path.exists('./models'):
        os.makedirs('./models')
    n_epochs, best_loss, step, early_stop = config['n_epochs'], math.inf,0,0
    for epoch in range(n_epochs):
        model.train()
        loss_record = []
        train_pbar = tqdm(train_loder,position=0,leave=True)

        for x,y in train_pbar:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred,y)
            loss.backward()
            optimizer.step()
            loss_record.append(loss.item())
            step += 1

            #展示当前进度
            train_pbar.set_description(f'Epoch {epoch+1}/{n_epochs}')
            train_pbar.set_postfix({'loss':loss.item()})

        mean_train_loss=np.mean(loss_record)
        writer.add_scalar('train_loss',mean_train_loss,step)

        #验证集
        model.eval()
        loss_record = []
        for x,y in valid_loder:
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred,y)
            loss_record.append(loss.item())
        
        mean_valid_loss=np.mean(loss_record)
        writer.add_scalar('valid_loss',mean_valid_loss,step)
        print(f'Epoch {epoch+1}/{n_epochs} Train Loss: {mean_train_loss:.4f} Valid Loss: {mean_valid_loss:.4f}')

        #Early Stopping
        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['model_path'])
            print(f'Epoch {epoch+1}/{n_epochs} Valid Loss: {mean_valid_loss:.4f} is the best, save model')
            early_stop = 0
        else:
            early_stop += 1
        if early_stop >= config['early_stop']:
            print('Early Stopping')
            break
    writer.close()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#配置
config = {
    'seed':42,
    'select_all':True,
    'valid_ratio':0.2,
    'n_epochs':3000,
    'batch_size':64,
    'learning_rate':0.001,
    'early_stop':400,
    'model_path':'./models/best_model.pth'
}
if __name__ == '__main__':
    #设置种子
    set_seed(config['seed'])
    #读取数据
    train_data,test_data=pd.read_csv('./data/train.csv').values ,pd.read_csv('./data/test.csv').values
    #分割训练集和验证集
    train_data,valid_data = train_valid_split(train_data,config['valid_ratio'],config['seed'])

    #挑选特征
    raw_x_train, raw_x_valid, raw_x_test, y_train, y_valid = select_feature(train_data,valid_data,test_data,config['select_all'])

    train_dataset = Covid19Dataset(raw_x_train,y_train)
    valid_dataset = Covid19Dataset(raw_x_valid,y_valid)


    #数据加载器
    train_loder = DataLoader(train_dataset,batch_size=config['batch_size'],shuffle=True,pin_memory=True)
    valid_loder = DataLoader(valid_dataset,batch_size=config['batch_size'],shuffle=False,pin_memory=True)


    model = mymodel(raw_x_train.shape[1]).to(device)
    train(train_loder,valid_loder,model,config,device)