import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import sys

# 将源码目录添加到路径中
SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SOURCE_DIR)
sys.path.append(SOURCE_DIR)

from model import BinaryVulnModel
from dataset import VulnDataset
from preprocess import CWE_TO_IDX

# --- 配置参数 ---
SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SOURCE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, '模型')
# 加载增强标签和全量特征缓存
TRAIN_AUGMENTED_CSV = os.path.join(DATA_DIR, 'augmented_train.csv')
STAGE2_CACHE_PATH = os.path.join(MODEL_DIR, 'stage2_features.pkl')
# 使用 Stage 1 产出的基础权重
CHECKPOINT_PATH = os.path.join(MODEL_DIR, 'base_model.pth')
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, 'final_model.pth')

BATCH_SIZE = 64
EPOCHS = 8
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-5

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.05, ignore_index=-1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        valid_mask = targets != self.ignore_index
        if not valid_mask.any():
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)
            
        inputs = inputs[valid_mask]
        targets = targets[valid_mask]
        log_pt = F.log_softmax(inputs, dim=1)
        pt = torch.exp(log_pt).gather(1, targets.unsqueeze(1)).squeeze(1)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss
        return focal_loss.mean()

def train_stage2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 加载数据
    print(f"Loading augmented labels from {TRAIN_AUGMENTED_CSV}...")
    df_extra = pd.read_csv(TRAIN_AUGMENTED_CSV)
    extra_label_map = df_extra.set_index('binary_id').to_dict('index')

    print(f"Loading cached features from {STAGE2_CACHE_PATH}...")
    with open(STAGE2_CACHE_PATH, 'rb') as f:
        full_cache = pickle.load(f)

    final_data = []
    cwe_targets = []
    for item in tqdm(full_cache, desc="Preprocessing"):
        bin_id = item['binary_id']
        if bin_id in extra_label_map:
            info = extra_label_map[bin_id]
            item['label'] = info['label']
            cwe_idx = CWE_TO_IDX.get(info.get('cwe_id', ""), -1)
            item['cwe_idx'] = cwe_idx
            # 归一化元数据
            item['meta'] = np.log1p(np.maximum(0, np.array(item['meta'])))
            final_data.append(item)
            if item['label'] == 1 and cwe_idx != -1:
                cwe_targets.append(cwe_idx)

    # 2. HEM: 自动化难样本挖掘
    print("\n========== Starting HEM (Hard Example Mining) Scan ==========")
    model = BinaryVulnModel(meta_dim=10).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    eval_loader = DataLoader(VulnDataset(preloaded_data=final_data, is_train=True), batch_size=BATCH_SIZE, shuffle=False)
    losses = []
    with torch.no_grad():
        for bytes_seq, meta, label, cwe_idx in tqdm(eval_loader, desc="Scanning Loss"):
            bytes_seq, meta, label, cwe_idx = bytes_seq.to(device), meta.to(device), label.to(device), cwe_idx.to(device)
            det_out, class_out = model(bytes_seq, meta)
            l = F.cross_entropy(det_out, label, reduction='none') + F.cross_entropy(class_out, cwe_idx, reduction='none', ignore_index=-1)
            losses.extend(l.cpu().numpy())

    # 挖掘 Top 300 难样本，并进行 10 倍重采样强化
    hard_indices = np.argsort(losses)[::-1][:300]
    boosted_samples = []
    for idx in hard_indices:
        for _ in range(10):
            boosted_samples.append(final_data[idx].copy())
    final_data.extend(boosted_samples)
    print(f"Identified 300 hard samples. Boosted training set to {len(final_data)} samples.")

    # 3. 准备训练
    class_counts = np.bincount(cwe_targets, minlength=86)
    alpha_weights = torch.FloatTensor(np.sqrt(len(cwe_targets) / (86.0 * (class_counts + 1)))).to(device)
    alpha_weights = alpha_weights / torch.mean(alpha_weights)

    dataset = VulnDataset(preloaded_data=final_data, is_train=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    criterion_det = nn.CrossEntropyLoss(label_smoothing=0.05)
    criterion_class = FocalLoss(alpha=alpha_weights, gamma=2.0, label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler('cuda')

    # 4. 冲刺训练
    print(f"\n========== Starting Stage 2 Refinement ({EPOCHS} Epochs) ==========")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for bytes_seq, meta, label, cwe_idx in pbar:
            bytes_seq, meta, label, cwe_idx = bytes_seq.to(device), meta.to(device), label.to(device), cwe_idx.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                det_out, class_out = model(bytes_seq, meta)
                loss = criterion_det(det_out, label) * 1.0 + criterion_class(class_out, cwe_idx) * 3.0
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        scheduler.step()

    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    print(f"Stage 2 complete! Final model saved to {FINAL_MODEL_PATH}")

if __name__ == "__main__":
    train_stage2()
