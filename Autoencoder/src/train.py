from pyexpat import model

import torch
from tqdm import tqdm
import os
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import random
import numpy as np
import torchvision.transforms as transforms


#设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


#定义模型
#1.全连接的Autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(64 * 64 * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 64 * 64 * 3),
            nn.Tanh(),
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

#2.卷积的Autoencoder
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh(),
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Vae(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.enc1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh(),
        )
    def encode(self, x):
        x = self.encoder(x)
        mu = self.enc1(x)
        logvar = self.enc2(x)
        return mu, logvar
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x = self.decoder(z)
        return x, mu, logvar

#定义vae的损失函数
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.MSELoss()(recon_x, x)
    KLD_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + KLD_loss


class PhotoDataset(Dataset):
    def __init__(self, tensors):
        self.tensors = torch.from_numpy(tensors) if isinstance(tensors, np.ndarray) else tensors
        if self.tensors.shape[-1] == 3:
            self.tensors = self.tensors.permute(0, 3, 1, 2)
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: x.to(torch.float32) / 255.),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    def __len__(self):
        return self.tensors.shape[0]
    def __getitem__(self, idx):
        x = self.tensors[idx]
        x = self.transform(x)
        return x

#定义超参数
config = {
    "seed": 42,
    "batch_size": 32,
    "num_epochs": 30,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_class": [Autoencoder(), ConvAutoencoder(), Vae()],
    "model_idx": 2,
}

if __name__ == '__main__':
    set_seed(config["seed"])
    #准备数据集
    train_data = np.load("../data/trainingset.npy")
    train_dataset = PhotoDataset(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    model = config["model_class"][config["model_idx"]].to(config["device"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    best_loss = float("inf")
    for epoch in range(config["num_epochs"]):
        model.train()
        train_loss = 0.0
        for x in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']}"):
            x = x.to(config["device"])
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            loss = vae_loss(recon_x, x, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        mean_train_loss = train_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{config['num_epochs']}, Train Loss: {mean_train_loss:.4f}")
        if mean_train_loss < best_loss:
            best_loss = mean_train_loss
            os.makedirs('../models', exist_ok=True)
            torch.save(model.state_dict(), '../models/autoencoder_best.pth')
            print("Best model saved!")
        else:
            print("No improvement, continue training.")
