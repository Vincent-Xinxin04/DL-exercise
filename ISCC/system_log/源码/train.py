import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import re
import math
from tqdm import tqdm
import os
import torch.nn.functional as F

# 修复 Windows 环境下的 OpenMP 报错
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 路径配置（统一使用相对路径）
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(SRC_DIR, 'data')
# 使用伪标签数据集
TRAIN_CSV = os.path.join(DATA_DIR, 'train_pseudo.csv')
MODEL_DIR = os.path.join(BASE_DIR, '模型')
VOCAB_PATH = os.path.join(MODEL_DIR, 'vocab.pth')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LINES = 180      # 每组日志的最大行数
MAX_LEN = 64         # 每行日志的最大单词数
BATCH_SIZE = 64      # 批大小
EPOCHS = 15          # 训练轮数
LR = 0.0005          # 学习率
IGNORE_INDEX = -1    # 损失计算中忽略的索引
# 伪标签数据缓存
CACHE_PATH = os.path.join(MODEL_DIR, 'full_data_cache_pseudo.pth')
NUM_FOLDS = 5        # 交叉验证折数

# 异常类型定义
ANOMALY_TYPES = [
    'none', 'timeout_retry', 'resource_exhaustion', 'slow_burn_warning',
    'state_conflict', 'parameter_drift', 'out_of_order', 'missing_step',
    'duplicate_event', 'cross_component_mismatch', 'partial_recovery_loop'
]
TYPE_TO_IDX = {t: i for i, t in enumerate(ANOMALY_TYPES)}
IDX_TO_TYPE = {i: t for i, t in enumerate(ANOMALY_TYPES)}

# FGM 对抗训练：在 Embedding 层加入扰动，提升模型鲁棒性
class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}
    def attack(self, epsilon=0.5, emb_name='embedding'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)
    def restore(self, emb_name='embedding'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# 日志预处理：清洗时间戳、IP、十六进制及ID，保留核心语义
def clean_log(text):
    text = re.sub(r'(INFO|WARN|ERROR|FATAL|DEBUG)', r' \1 ', text, flags=re.IGNORECASE)
    text = re.sub(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', '<TIME>', text)
    text = re.sub(r'\d+\.\d+\.\d+\.\d+', '<IP>', text)
    text = re.sub(r'0x[0-9a-fA-F]{5,}', '<HEX>', text)
    text = re.sub(r'seg_[0-9a-fA-F]+', 'seg_<ID>', text)
    text = re.sub(r'id_[0-9a-fA-F]+', 'id_<ID>', text)
    text = re.sub(r'([\[\]\(\)\:\=\,])', r' \1 ', text)
    return text.lower()

# 正余弦位置编码：为 Transformer 提供行位置信息
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x): return x + self.pe[:, :x.size(1), :]

# 日志数据集加载类
class LogDataset(Dataset):
    def __init__(self, df=None, vocab=None, is_test=False, preprocessed=None):
        self.is_test = is_test
        self.vocab = vocab or {'<PAD>': 0, '<UNK>': 1}
        if preprocessed: self.data, self.labels = preprocessed
        else:
            self.data, self.labels = [], []
            self._prepare(df)
    def _prepare(self, df):
        for _, row in tqdm(df.iterrows(), total=len(df), desc="准备数据"):
            log_text = str(row['log_text']).split('\n')
            processed_lines = []
            for line in log_text[:MAX_LINES]:
                line = clean_log(line)
                words = re.findall(r'\w+|[\[\]\(\)\:\=\,]', line)
                word_ids = [self.vocab.get(w, self.vocab['<UNK>'] if self.is_test else (self.vocab.setdefault(w, len(self.vocab)))) for w in words[:MAX_LEN]]
                word_ids += [0] * (MAX_LEN - len(word_ids))
                processed_lines.append(word_ids)
            num_actual = len(processed_lines)
            processed_lines += [[0] * MAX_LEN] * (MAX_LINES - num_actual)
            self.data.append(processed_lines)
            if not self.is_test:
                line_labels = [0] * MAX_LINES
                for i in range(num_actual, MAX_LINES): line_labels[i] = IGNORE_INDEX
                if row['has_anomaly'] == 1:
                    for span in str(row['all_spans']).split(';'):
                        try:
                            s, e, t = span.split('|')
                            for i in range(max(0, int(s)), min(num_actual, int(e) + 1)): line_labels[i] = TYPE_TO_IDX.get(t, 0)
                        except: continue
                self.labels.append(line_labels)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.long)
        if self.is_test or not self.labels: return x
        return x, torch.tensor(self.labels[idx], dtype=torch.long)

# 核心模型：多尺度 CNN + 双向 LSTM + Transformer
class LogModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_classes=11):
        super(LogModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # 多尺度卷积核提取不同粒度的特征
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, 48, k, padding=k//2) for k in [1, 3, 5, 7]])
        self.lstm = nn.LSTM(48 * 4, hidden_dim, batch_first=True, bidirectional=True, num_layers=2, dropout=0.3)
        self.pos_encoder = PositionalEncoding(hidden_dim * 2, MAX_LINES)
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim * 2, nhead=8, dim_feedforward=hidden_dim * 4, dropout=0.3, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    @staticmethod
    def post_process_spans(spans):
        # 0.952 基准策略：过滤长度小于 2 的异常区间
        filtered = [s for s in spans if (s['end'] - s['start'] + 1) >= 2]
        if not filtered: return []
        merged = [filtered[0]]
        for cur in filtered[1:]:
            prev = merged[-1]
            # 合并间距小于等于 2 的同类型异常
            if cur['type'] == prev['type'] and cur['start'] - prev['end'] <= 2: prev['end'] = cur['end']
            else: merged.append(cur)
        return merged

    def forward(self, x):
        b, s, l = x.size()
        x = x.view(b * s, l)
        embeds = self.embedding(x).transpose(1, 2)
        line_embeds = torch.cat([F.relu(conv(embeds)).max(dim=2)[0] for conv in self.convs], dim=1).view(b, s, -1)
        lstm_out, _ = self.lstm(line_embeds)
        transformer_in = self.pos_encoder(lstm_out)
        transformer_out = self.transformer(transformer_in)
        return self.classifier(transformer_out)

def train():
    print(f"当前训练设备: {DEVICE}")
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = pd.read_csv(TRAIN_CSV)
    if os.path.exists(CACHE_PATH): 
        all_data, all_labels, vocab = torch.load(CACHE_PATH)
    else:
        full_dataset = LogDataset(df)
        all_data, all_labels, vocab = full_dataset.data, full_dataset.labels, full_dataset.vocab
        torch.save((all_data, all_labels, vocab), CACHE_PATH)
    torch.save(vocab, VOCAB_PATH)
    all_data, all_labels = np.array(all_data), np.array(all_labels)
    kf = KFold(5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kf.split(all_data)):
        print(f"\n--- Fold {fold+1}/5 ---")
        train_loader = DataLoader(LogDataset(preprocessed=(all_data[train_idx].tolist(), all_labels[train_idx].tolist()), vocab=vocab), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(LogDataset(preprocessed=(all_data[val_idx].tolist(), all_labels[val_idx].tolist()), vocab=vocab), batch_size=BATCH_SIZE)
        model = LogModel(len(vocab)).to(DEVICE)
        fgm = FGM(model)
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        # 类别权重：降低正常类的权重，让模型更关注异常
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.5]+[1.0]*10, device=DEVICE), ignore_index=IGNORE_INDEX, label_smoothing=0.1)
        scaler = torch.amp.GradScaler('cuda')
        best_f1, fold_path = 0, os.path.join(MODEL_DIR, f'model_fold_{fold}_final.pth')
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            for x, y in tqdm(train_loader, desc=f"轮次 {epoch+1}"):
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    loss = criterion(model(x).view(-1, 11), y.view(-1))
                scaler.scale(loss).backward()
                # 对抗攻击
                fgm.attack()
                with torch.amp.autocast('cuda'):
                    loss_adv = criterion(model(x).view(-1, 11), y.view(-1))
                scaler.scale(loss_adv).backward()
                fgm.restore() # 恢复参数
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()
            scheduler.step()
            model.eval()
            all_p, all_t = [], []
            with torch.no_grad():
                for x, y in val_loader:
                    preds = model(x.to(DEVICE)).argmax(-1).view(-1).cpu().numpy()
                    labels = y.view(-1).numpy()
                    mask = labels != IGNORE_INDEX
                    all_p.extend(preds[mask]); all_t.extend(labels[mask])
            f1 = f1_score(all_t, all_p, average='macro')
            print(f"轮次 {epoch+1} 损失: {total_loss/len(train_loader):.4f} 验证集 F1: {f1:.4f}")
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), fold_path)
                print(f"模型已保存 (F1: {f1:.4f})")

if __name__ == "__main__":
    train()
