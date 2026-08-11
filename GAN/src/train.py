import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
from torchvision import transforms
import torchvision
import torch.nn as nn



#设置随机种子
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

#定义一个数据集
class PhotoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = os.listdir(root_dir)
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        return img

#定义GAN神经网络
class Generator(nn.Module):
    def __init__(self,input_dim,feature_dim = 64):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(input_dim, feature_dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(feature_dim * 8 * 4 * 4),
            nn.ReLU()
        )
        self.l2 = nn.Sequential(
            self.deconv_bn_relu(feature_dim * 8 ,feature_dim * 4),
            self.deconv_bn_relu(feature_dim * 4,feature_dim * 2),
            self.deconv_bn_relu(feature_dim * 2,feature_dim * 1),
        )
        self.l3 = nn.Sequential(
            nn.ConvTranspose2d(feature_dim * 1, 3, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            nn.Tanh()
        )

    def deconv_bn_relu(self,in_channels,out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self,x):
        x = self.l1(x)
        x = x.view(x.size(0),-1,4,4)
        x = self.l2(x)
        x = self.l3(x)
        return x

class Discriminator(nn.Module):
    def __init__(self,input_dim,feature_dim = 64):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Conv2d(input_dim, feature_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.l2 = nn.Sequential(
            self.conv_bn_relu(feature_dim,feature_dim * 2),
            self.conv_bn_relu(feature_dim * 2,feature_dim * 4),
            self.conv_bn_relu(feature_dim * 4,feature_dim * 8),

        )
        self.l3 = nn.Sequential(
            nn.Conv2d(feature_dim * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
    def conv_bn_relu(self,in_channels,out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self,x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = x.view(x.size(0),-1)
        x = x.squeeze()
        return x

#定义transform
transform = transforms.Compose([
    transforms.RandomVerticalFlip(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

#定义超参数
batch_size = 64
num_epochs = 15
learning_rate = 0.0002
beta1 = 0.5
beta2 = 0.999


if __name__ == '__main__':
    #加载数据集
    train_dataset = PhotoDataset(root_dir='../data/faces', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    #初始化模型、损失函数、优化器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = Generator(input_dim=100).to(device)
    D = Discriminator(input_dim=3).to(device)
    criterion = nn.BCELoss()

    opt_G = torch.optim.Adam(G.parameters(), lr=learning_rate, betas=(beta1, beta2))
    opt_D = torch.optim.Adam(D.parameters(), lr=learning_rate, betas=(beta1, beta2))

    #用于生成固定噪声来可视化训练进度
    fixed_noise = torch.randn(16, 100, device=device)

    #训练循环
    os.makedirs('../models', exist_ok=True)
    os.makedirs('../output', exist_ok=True)

    print(f'Using device: {device}')
    print(f'Train samples: {len(train_dataset)}')

    for epoch in range(num_epochs):
        G_loss_total = 0.0
        D_loss_total = 0.0

        for real_imgs in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            real_imgs = real_imgs.to(device)
            batch_size_curr = real_imgs.size(0)

            #标签平滑：真实图用 0.9 而非 1.0，防止 D 过度自信
            real_labels = torch.full((batch_size_curr,), 0.9, device=device)
            fake_labels = torch.zeros(batch_size_curr, device=device)

            #训练 Discriminator
            #真实图：D 应输出接近 1
            real_out = D(real_imgs)
            loss_real = criterion(real_out, real_labels)

            #生成图：D 应输出接近 0
            noise = torch.randn(batch_size_curr, 100, device=device)
            fake_imgs = G(noise)
            fake_out = D(fake_imgs.detach())
            loss_fake = criterion(fake_out, fake_labels)

            D_loss = loss_real + loss_fake
            opt_D.zero_grad()
            D_loss.backward()
            opt_D.step()

            #训练 Generator（训两次，给 G 更多机会追赶 D）
            for _ in range(2):
                noise = torch.randn(batch_size_curr, 100, device=device)
                fake_imgs = G(noise)
                fake_out = D(fake_imgs)
                G_loss = criterion(fake_out, real_labels)

                opt_G.zero_grad()
                G_loss.backward()
                opt_G.step()

            G_loss_total += G_loss.item()
            D_loss_total += D_loss.item()

        mean_G_loss = G_loss_total / len(train_loader)
        mean_D_loss = D_loss_total / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs} | G Loss: {mean_G_loss:.4f} | D Loss: {mean_D_loss:.4f}')

        #每 5 轮保存生成样本 + 模型权重
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                gen_imgs = G(fixed_noise).cpu()
            grid = torchvision.utils.make_grid(gen_imgs, nrow=4, normalize=True, value_range=(-1, 1))
            torchvision.utils.save_image(grid, f'../output/epoch_{epoch+1:03d}.png')
            torch.save(G.state_dict(), f'../models/G_epoch_{epoch+1}.pth')
            torch.save(D.state_dict(), f'../models/D_epoch_{epoch+1}.pth')
            print(f'  -> Saved samples & models for epoch {epoch+1}')

    torch.save(G.state_dict(), '../models/G_final.pth')
    torch.save(D.state_dict(), '../models/D_final.pth')
    print('Training completed!')