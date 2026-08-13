from train import config, PhotoDataset, set_seed
import torch
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import csv


if __name__ == '__main__':
    #设置随机种子
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    #加载数据
    print('Loading test data...')
    test_data = np.load('../data/testingset.npy')
    test_dataset = PhotoDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    #加载模型
    model = config["model_class"][config["model_idx"]].to(device)
    model_path = '../models/autoencoder_best.pth'
    if not os.path.exists(model_path):
        print(f'Model not found: {model_path}')
        print('Please run train.py first to train the model.')
        exit(1)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print('Model loaded successfully!')

    #推理：计算重建误差作为异常得分
    print('Running inference...')
    scores = []

    with torch.no_grad():
        for x in tqdm(test_loader):
            x = x.to(device)
            recon_x, mu, logvar = model(x)
            #每张图所有像素的均方误差作为异常得分
            error = ((recon_x - x) ** 2).mean(dim=[1, 2, 3])
            scores.extend(error.cpu().numpy().tolist())

    #保存结果
    os.makedirs('../data', exist_ok=True)
    output_path = '../data/anomaly_scores.csv'
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'score'])
        for i, score in enumerate(scores):
            writer.writerow([i, float(score)])

    print(f'Saved {len(scores)} anomaly scores to {output_path}')
