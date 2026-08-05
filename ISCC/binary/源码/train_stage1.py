import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.model_selection")

import sys

# 将源码目录添加到路径中
SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SOURCE_DIR)
sys.path.append(SOURCE_DIR)

from model import BinaryVulnModel
from dataset import VulnDataset

# Config
DATA_DIR = os.path.join(SOURCE_DIR, 'data')
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
BIN_DIR = os.path.join(DATA_DIR, 'binaries', 'binaries')
MODEL_DIR = os.path.join(BASE_DIR, '模型')
CACHE_PATH = os.path.join(MODEL_DIR, 'features_cache.pkl')
BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 0.001
SEQ_LEN = 4096

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.05, ignore_index=-100):
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

def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Dataset
    print("Loading dataset...")
    full_dataset = VulnDataset(TRAIN_CSV, BIN_DIR, seq_len=SEQ_LEN, is_train=True, cache_path=CACHE_PATH)
    
    if full_dataset.use_cache:
        print("Normalizing meta features...")
        for i in range(len(full_dataset.data)):
            full_dataset.data[i]['meta'] = np.log1p(np.maximum(0, full_dataset.data[i]['meta']))

    # 2. Calculate Class Weights
    from preprocess import CWE_TO_IDX
    cwe_targets = []
    for i in range(len(full_dataset)):
        item = full_dataset.data[i] if full_dataset.use_cache else full_dataset[i]
        label = item['label'] if full_dataset.use_cache else item[2]
        cwe_idx = item['cwe_idx'] if full_dataset.use_cache else item[3]
        if isinstance(cwe_idx, str): cwe_idx = CWE_TO_IDX.get(cwe_idx, -1)
        if label == 1 and cwe_idx != -1:
            cwe_targets.append(cwe_idx)

    class_counts = np.bincount(cwe_targets, minlength=86)
    alpha_weights = torch.FloatTensor(np.sqrt(len(cwe_targets) / (86.0 * (class_counts + 1)))).to(device)
    alpha_weights = alpha_weights / torch.mean(alpha_weights)
    
    # 3. Simple Train/Val Split (Single Model)
    print("\n========== Starting Single Model Training (No K-Fold) ==========")
    indices = np.arange(len(full_dataset))
    train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=42, stratify=None)
    
    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)

    model = BinaryVulnModel(meta_dim=10).to(device)
    criterion_det = nn.CrossEntropyLoss(label_smoothing=0.05)
    criterion_class = FocalLoss(alpha=alpha_weights, gamma=2.0, label_smoothing=0.05, ignore_index=-1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    scaler = torch.amp.GradScaler('cuda')

    best_f1 = 0
    best_model_path = os.path.join(MODEL_DIR, 'base_model.pth')
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for bytes_seq, meta, label, cwe_idx in pbar:
            bytes_seq, meta, label, cwe_idx = bytes_seq.to(device), meta.to(device), label.to(device), cwe_idx.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                det_out, class_out = model(bytes_seq, meta)
                loss = criterion_det(det_out, label) * 1.5 + criterion_class(class_out, cwe_idx)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{scheduler.get_last_lr()[0]:.5f}"})

        # Validation
        model.eval()
        all_det_preds, all_det_labels = [], []
        all_class_preds, all_class_labels = [], []
        with torch.no_grad():
            for bytes_seq, meta, label, cwe_idx in val_loader:
                bytes_seq, meta = bytes_seq.to(device), meta.to(device)
                with torch.amp.autocast('cuda'):
                    det_out, class_out = model(bytes_seq, meta)
                all_det_preds.extend(torch.argmax(det_out, dim=1).cpu().numpy())
                all_det_labels.extend(label.numpy())
                mask = (cwe_idx != -1).numpy()
                if mask.any():
                    all_class_preds.extend(torch.argmax(class_out, dim=1).cpu().numpy()[mask])
                    all_class_labels.extend(cwe_idx.numpy()[mask])

        f1_det = f1_score(all_det_labels, all_det_preds, average='macro')
        f1_class = f1_score(all_class_labels, all_class_preds, average='macro', zero_division=0) if len(all_class_labels)>0 else 0
        avg_f1 = (f1_det + f1_class) / 2
        print(f"Epoch {epoch+1}: Det-F1: {f1_det:.4f}, Class-F1: {f1_class:.4f}, Avg-F1: {avg_f1:.4f}")

        if avg_f1 > best_f1:
            best_f1 = avg_f1
            torch.save(model.state_dict(), best_model_path)
            print(f"---> Saved Best Model (Avg-F1: {best_f1:.4f})")
        
        scheduler.step()
            
    print(f"\nTraining Complete. Best Avg-F1: {best_f1:.4f}")

if __name__ == "__main__":
    train()
