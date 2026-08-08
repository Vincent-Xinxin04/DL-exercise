from torch.utils.data import random_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import math
from tqdm import tqdm
from pathlib import Path
#设置随机种子，有助于复现
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True  #使用确定性算法进行卷积操作
    torch.backends.cudnn.benchmark = False  #关闭自动算法搜索
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# Warmup + Cosine 学习率调度
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

#数据集类
class Voicedataset(Dataset):
    def __init__(self, data_path, seg_len=1024):
        super().__init__()
        mapping_path = Path(data_path) / 'mapping.json'
        self.speaker2id = json.load(open(mapping_path))['speaker2id']

        metadata_path = Path(data_path) / 'metadata.json'
        self.metadata = json.load(open(metadata_path))['speakers']
        self.data = []
        self.seg_len = seg_len
        self.data_path = data_path
        self.speaker_num = len(self.metadata.keys())
        for speaker in self.metadata:
            for pkg in self.metadata[speaker]:
                self.data.append([pkg['feature_path'], self.speaker2id[speaker]])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        feature_path,speaker_id = self.data[idx]

        feature = torch.load(os.path.join(self.data_path, feature_path))

        if len(feature) > self.seg_len:
            start = random.randint(0, len(feature) - self.seg_len)
            feature = feature[start:start+self.seg_len,:]
        else:
            feature = F.pad(feature, (0, 0, 0, self.seg_len - len(feature)))
        
        return feature, speaker_id
                
    def get_speaker_num(self):
        return self.speaker_num

#定义模型
class TransformerModel(nn.Module):
    def __init__(self,d_model=80,n_spkgs=600,dropout=0.1):
        super().__init__()
        self.prenet = nn.Linear(40,d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=8, dim_feedforward=1024, dropout=dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_spkgs)
        )

    def forward(self, x):
        x = self.prenet(x)
        x = x.permute(1, 0, 2)
        x = self.encoder(x)  # 计算自注意力
        x = x.permute(1, 0, 2)
        x = x.mean(dim=1)
        output = self.classifier(x)
        return output

#切分训练集和验证集
def split_train_val(dataset, val_ratio=0.2):
    val_len = int(len(dataset) * val_ratio)
    train_len = len(dataset) - val_len
    lengths = [train_len, val_len]
    train_dataset, val_dataset = random_split(dataset, lengths)
    return train_dataset, val_dataset


#设置一些超参数
batch_size = 32
num_epochs = 15
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
early_stop = 0
if __name__ == '__main__':
    set_seed(42)

    print(f'Device: {device}')
    #加载数据集
    print('Loading dataset...')
    train_dataset, val_dataset = split_train_val(Voicedataset('../data/Dataset', seg_len=256))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(f'Train samples: {len(train_dataset)}, Valid samples: {len(val_dataset)}')
    print(f'Speakers: {Voicedataset("../data/Dataset").get_speaker_num()}')
    
    os.makedirs('../models', exist_ok=True)
    best_acc = 0.0
    best_model = None
    model = TransformerModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    total_steps = num_epochs * len(train_loader)
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print('Starting training...')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        val_loss = 0.0
        train_acc = 0.0
        val_acc = 0.0
        for x,y in tqdm(train_loader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            train_acc += (output.argmax(dim=1) == y).sum().item()
        mean_train_loss = train_loss / len(train_loader)
        mean_train_acc = train_acc / len(train_dataset)

        model.eval()
        for x,y in tqdm(val_loader):
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                output = model(x)
                loss = criterion(output, y)
                val_loss += loss.item()
                val_acc += (output.argmax(dim=1) == y).sum().item()
        mean_val_loss = val_loss / len(val_loader)
        mean_val_acc = val_acc / len(val_dataset)

        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'  Train Loss: {mean_train_loss:.4f}, Train Acc: {mean_train_acc:.4f}')
        print(f'  Valid Loss: {mean_val_loss:.4f}, Valid Acc: {mean_val_acc:.4f}')

        if mean_val_acc > best_acc:
            best_acc = mean_val_acc
            best_model = model.state_dict()
            torch.save(best_model, '../models/best_model.pth')
            print(f'  Best model saved! (Val Acc: {best_acc:.4f})')
            early_stop = 0
        else:
            early_stop += 1
            if early_stop == 3:
                print('Early stopping...')
                break

    print(f'Training completed! Best Val Acc: {best_acc:.4f}')
