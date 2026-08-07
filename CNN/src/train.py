import torch
import random
import math
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
#设置CNN的种子
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

#定义一些transform用于数据增强
train_tfm = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

test_tfm = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

#定义food数据集
class FoodDataset(Dataset):
    def __init__(self, path, transform=test_tfm,mode='train'):
        super().__init__()
        self.path = path
        self.transform = transform
        self.mode = mode
        self.data = sorted([os.path.join(path,f) for f in os.listdir(path)])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = Image.open(self.data[idx])
        img = self.transform(img)
        if self.mode == 'train' or self.mode == 'valid':
            label = int(os.path.basename(self.data[idx]).split('_')[0])
            return img, label
        else:
            return img

#定义CNN神经网络
class FoodModel(nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),       # (64,  224, 224)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),               # (64,  112, 112)

            nn.Conv2d(64, 128, 3, 1, 1),      # (128, 112, 112)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),               # (128,  56,  56)

            nn.Conv2d(128, 256, 3, 1, 1),     # (256,  56,  56)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),               # (256,  28,  28)

            nn.Conv2d(256, 512, 3, 1, 1),     # (512,  28,  28)
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),               # (512,  14,  14)

            nn.Conv2d(512, 512, 3, 1, 1),     # (512,  14,  14)
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),               # (512,   7,   7)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                       # 512 * 7 * 7 = 25088
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.classifier(x)
        return x


#设置一些超参数
set_seed(42)
batch_size = 64
learning_rate = 0.001
n_epochs = 10

#训练数据集准备
train_set = FoodDataset('../data/food11/training', transform=train_tfm,mode='train')
valid_set = FoodDataset('../data/food11/validation', transform=test_tfm,mode='valid')
train_loder = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_loder = DataLoader(valid_set, batch_size=batch_size, shuffle=False)

#训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    if not os.path.exists('../models'):
        os.makedirs('../models', exist_ok=True)

    print(f'Train samples: {len(train_set)}, Valid samples: {len(valid_set)}')
    print(f'Device: {device}')

    model = FoodModel().to(device)
    print('Starting training...')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    best_loss = math.inf
    early_stop = 0

    for epoch in range(n_epochs):
        train_acc = 0.0
        valid_acc = 0.0
        train_loss = 0.0
        valid_loss = 0.0
        model.train()
        for x,y in tqdm(train_loder):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred,y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += (pred.argmax(1) == y).sum().item()

        mean_train_loss = train_loss / len(train_loder)
        mean_train_acc = train_acc / len(train_set)
        #展示当前进度
        print(f'Epoch {epoch+1}/{n_epochs}')
        print(f'Train Loss: {mean_train_loss:.4f}, Train Acc: {mean_train_acc:.4f}')

        #验证集
        model.eval()
        for x,y in valid_loder:
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred,y)
            valid_loss += loss.item()
            valid_acc += (pred.argmax(1) == y).sum().item()
        mean_valid_loss = valid_loss / len(valid_loder)
        mean_valid_acc = valid_acc / len(valid_set)
        #展示当前进度
        print(f'Valid Loss: {mean_valid_loss:.4f}, Valid Acc: {mean_valid_acc:.4f}')

        #保存模型
        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), '../models/best_model.pth')
            print('Best model saved!')
            early_stop = 0
        else:
            early_stop += 1
            if early_stop == 3:
                print('Early stop!')
                break
    print('Training completed!')
