import torch
import torch.nn as nn
import torchvision
import os
import random
import numpy as np
from train import Generator, set_seed

#设置随机种子
set_seed(42)

#超参数（需与训练时一致）
feature_dim = 64
input_dim = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#初始化 Generator 并加载权重
G = Generator(input_dim=input_dim, feature_dim=feature_dim).to(device)
model_path = '../models/G_final.pth'
if not os.path.exists(model_path):
    print(f'Model not found: {model_path}')
    print('Please run train.py first to train the model.')
    exit(1)

G.load_state_dict(torch.load(model_path, map_location=device))
G.eval()
print(f'Model loaded from {model_path}')

#生成图片
os.makedirs('../output', exist_ok=True)

num_images = 64
with torch.no_grad():
    noise = torch.randn(num_images, input_dim, device=device)
    gen_imgs = G(noise).cpu()

#保存为网格图
grid = torchvision.utils.make_grid(gen_imgs, nrow=8, normalize=True, value_range=(-1, 1))
torchvision.utils.save_image(grid, '../output/inference_result.png')
print(f'Saved {num_images} generated images to ../output/inference_result.png')
