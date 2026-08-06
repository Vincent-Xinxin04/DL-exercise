import os
import random
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import math
def load_feat(path):
    feat = torch.load(path)
    return feat

#做语音处理的时候需要进行平移获取上下文语义
def shift(x,n):
    if n < 0:
        left = x[0].repeat(-n,1)
        right = x[:n]
    elif n > 0:
        right = x[-1].repeat(n,1)
        left = x[n:]
    else:
        return x
    return torch.cat([left,right],dim=0)

#拼接上下文的帧
def concat_feat(x,concat_n):
    assert concat_n % 2 == 1
    if concat_n < 2 :
        return x
    seq_len, feature_dim = x.size(0), x.size(1)
    x = x.repeat(1,concat_n)
    x = x.view(seq_len,concat_n,feature_dim).permute(1,0,2)
    mid = concat_n // 2
    for r_idx in range(1,mid+1):
        x[mid+r_idx, :] = shift(x[mid+r_idx],r_idx)
        x[mid-r_idx, :] = shift(x[mid-r_idx],-r_idx)
    return x.permute(1,0,2).view(seq_len,concat_n*feature_dim)

#数据预处理
def preprocess_data(split,feat_dir,phone_path,concat_nframes,train_ratio=0.8,train_valid_seed=1337):
    mode = 'train' if (split == 'train' or split == 'val') else 'test'

    label_dict = {}
    if mode!='test':
        phone_file = open(os.path.join(phone_path,mode+'_labels.txt')).readlines()

        for line in phone_file:
            line = line.strip().split(' ')
            label_dict[line[0]] = [int(p) for p in line[1:]]
    if split == 'train' or split == 'val':
        usage_list = open(os.path.join(phone_path,'train_split.txt')).readlines()
        random.seed(train_valid_seed)
        random.shuffle(usage_list)
        percent = int(len(usage_list) * train_ratio)
        usage_list = usage_list[:percent] if split == 'train' else usage_list[percent:]
    
    elif split == 'test':
        usage_list = open(os.path.join(phone_path,'test_split.txt')).readlines()

    usage_list = [line.strip('\n') for line in usage_list]

    max_len = int(3e6)

    X = torch.empty(max_len,39*concat_nframes)
    if mode!='test':
        y = torch.empty(max_len,dtype=torch.long)

    idx = 0
    for i, fname in tqdm(enumerate(usage_list)):
        feat = load_feat(os.path.join(feat_dir,mode,fname+'.pt'))
        cur_len = len(feat)
        feat = concat_feat(feat,concat_nframes)
        if mode!='test':
            label = torch.LongTensor(label_dict[fname])

        X[idx: idx+cur_len, :] = feat
        if mode!='test':
            y[idx: idx+cur_len] = label 
        idx += cur_len
    X = X[:idx]
    if mode!='test':
        y = y[:idx]
    
    if mode!='test':
        return X,y
    else: 
        return X


#定义数据集
class VoiceDataset(Dataset):
    def __init__(self,x,y=None):
        self.x = x
        self.y = y
    def __len__(self):
        return self.x.size(0)
    def __getitem__(self,idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

#定义分类模型
class VoiceModel(nn.Module):
    def __init__(self,input_dim,output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),

        )
    def forward(self, x):
        return self.fc(x)


#设置种子
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


#定义一些超参数
concat_nframes = 3
train_ratio = 0.8
seed = 42
batch_size = 128
num_epochs = 10
learning_rate = 0.001
model_path = os.path.join('../models','voice_model_best.pth')
input_dim = 39*concat_nframes
output_dim = 41
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    print('Using device:', device)

    if not os.path.exists('../models'):
        os.makedirs('../models', exist_ok=True)

    #准备数据集
    train_x,train_y = preprocess_data(split='train',feat_dir='../data/libriphone/feat',phone_path='../data/libriphone',concat_nframes=concat_nframes,train_ratio=train_ratio)
    val_x,val_y = preprocess_data(split='val',feat_dir='../data/libriphone/feat',phone_path='../data/libriphone',concat_nframes=concat_nframes,train_ratio=train_ratio)


    train_set = VoiceDataset(train_x,train_y)
    val_set = VoiceDataset(val_x,val_y)

    train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(val_set,batch_size=batch_size,shuffle=True)


    set_seed(seed)

    #初始化模型
    model = VoiceModel(input_dim,output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    #训练
    best_acc = 0
    early_stop = 0
    for epoch in range(num_epochs):
        train_acc = 0.0
        val_acc = 0.0
        train_loss = 0.0
        val_loss = 0.0

        model.train()
        for x,y in tqdm(train_loader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_acc += (predicted == y).sum().item()

        mean_train_loss = train_loss / len(train_loader)
        mean_train_acc = train_acc / len(train_set)

        model.eval()
        for x,y in tqdm(val_loader):
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_acc += (predicted == y).sum().item()
        mean_val_loss = val_loss / len(val_loader)
        mean_val_acc = val_acc / len(val_set)

        if mean_val_acc > best_acc:
            best_acc = mean_val_acc
            torch.save(model.state_dict(), model_path)
            early_stop = 0
        else: 
            early_stop += 1
            if early_stop == 3:
                print('Early Stopping')                
                break