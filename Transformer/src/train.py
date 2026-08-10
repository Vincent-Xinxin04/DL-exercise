import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import random
import os
import re
import pickle
from tqdm import tqdm
from collections import Counter
import numpy as np

# ==================== 特殊标记 ====================
PAD_TOKEN = '<pad>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
UNK_TOKEN = '<unk>'

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

# ==================== 设置随机种子 ====================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# ==================== 分词 ====================
def tokenize_en(text):
    text = text.lower().strip()
    text = re.sub(r'([.,!?;:\"\'()\[\]{}])', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.split()

def tokenize_zh(text):
    text = text.strip()
    tokens = []
    for ch in text:
        if ch == ' ':
            continue
        tokens.append(ch)
    return tokens

# ==================== 构建词表 ====================
def build_vocab(sentences, max_size, tokenize_fn):
    counter = Counter()
    for sent in sentences:
        tokens = tokenize_fn(sent)
        counter.update(tokens)

    vocab = {
        PAD_TOKEN: PAD_IDX,
        SOS_TOKEN: SOS_IDX,
        EOS_TOKEN: EOS_IDX,
        UNK_TOKEN: UNK_IDX
    }
    for token, _ in counter.most_common(max_size - 4):
        vocab[token] = len(vocab)
    return vocab

def encode(tokens, vocab):
    return [vocab.get(t, UNK_IDX) for t in tokens]

# ==================== 数据集 ====================
class TranslationDataset(Dataset):
    def __init__(self, en_path, zh_path, en_vocab, zh_vocab, max_len=50, max_samples=None):
        self.max_len = max_len
        self.en_vocab = en_vocab
        self.zh_vocab = zh_vocab
        self.data = []

        with open(en_path, 'r', encoding='utf-8') as f_en, \
             open(zh_path, 'r', encoding='utf-8') as f_zh:
            for i, (en_line, zh_line) in enumerate(zip(f_en, f_zh)):
                if max_samples is not None and i >= max_samples:
                    break
                en_tokens = tokenize_en(en_line)
                zh_tokens = tokenize_zh(zh_line)
                if len(en_tokens) > 0 and len(zh_tokens) > 0:
                    en_ids = encode(en_tokens, en_vocab)[:max_len]
                    zh_ids = encode(zh_tokens, zh_vocab)[:max_len]
                    self.data.append((en_ids, zh_ids))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        en_ids, zh_ids = self.data[idx]
        enc_input = [SOS_IDX] + en_ids + [EOS_IDX]
        dec_input = [SOS_IDX] + zh_ids
        dec_target = zh_ids + [EOS_IDX]

        return torch.tensor(enc_input), torch.tensor(dec_input), torch.tensor(dec_target)

def collate_fn(batch):
    enc_inputs, dec_inputs, dec_targets = zip(*batch)

    enc_inputs = nn.utils.rnn.pad_sequence(enc_inputs, batch_first=True, padding_value=PAD_IDX)
    dec_inputs = nn.utils.rnn.pad_sequence(dec_inputs, batch_first=True, padding_value=PAD_IDX)
    dec_targets = nn.utils.rnn.pad_sequence(dec_targets, batch_first=True, padding_value=PAD_IDX)

    return enc_inputs, dec_inputs, dec_targets

# ==================== 位置编码 ====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# ==================== 缩放点积注意力 ====================
def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn = F.softmax(scores, dim=-1)
    output = torch.matmul(attn, v)
    return output

# ==================== 多头注意力 ====================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.d_k = d_model // nhead
        self.nhead = nhead

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q = self.w_q(q).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)

        attn_output = scaled_dot_product_attention(q, k, v, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.nhead * self.d_k)

        output = self.w_o(attn_output)
        return output

# ==================== 前馈网络 ====================
class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

# ==================== Encoder Layer ====================
class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.ff = FeedForward(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        attn_out = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout1(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout2(ff_out))
        return x

# ==================== Decoder Layer ====================
class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.ff = FeedForward(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_out))
        cross_out = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout2(cross_out))
        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout3(ff_out))
        return x

# ==================== Transformer NMT 模型 ====================
class TransformerNMT(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, nhead=8,
                 num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=512,
                 dropout=0.1, max_len=5000):
        super().__init__()

        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=PAD_IDX)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])

        self.output_fc = nn.Linear(d_model, tgt_vocab_size)
        self.d_model = d_model

    def _generate_square_subsequent_mask(self, sz, device):
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, 0).masked_fill(mask == 0, 1)
        return mask

    def _create_padding_mask(self, x):
        return (x != PAD_IDX).unsqueeze(1).unsqueeze(2)

    def encode(self, src):
        src_mask = self._create_padding_mask(src)
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x, src_mask

    def decode(self, tgt, enc_output, src_mask):
        tgt_mask = self._generate_square_subsequent_mask(tgt.size(1), tgt.device)
        tgt_pad_mask = self._create_padding_mask(tgt)
        tgt_mask = tgt_mask * tgt_pad_mask

        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        for layer in self.decoder_layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return x

    def forward(self, src, tgt):
        enc_output, src_mask = self.encode(src)
        dec_output = self.decode(tgt, enc_output, src_mask)
        output = self.output_fc(dec_output)
        return output

    def greedy_decode(self, src, max_len=50):
        self.eval()
        enc_output, src_mask = self.encode(src)

        batch_size = src.size(0)
        ys = torch.tensor([[SOS_IDX]] * batch_size, device=src.device)

        for _ in range(max_len):
            dec_output = self.decode(ys, enc_output, src_mask)
            logits = self.output_fc(dec_output[:, -1:, :])
            next_token = logits.argmax(dim=-1)
            ys = torch.cat([ys, next_token], dim=1)

            if (next_token == EOS_IDX).all():
                break

        return ys

# ==================== Warmup + Cosine 学习率调度 ====================
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ==================== 超参数 ====================
MAX_LEN = 50
MAX_TRAIN_SAMPLES = 150000
EN_VOCAB_SIZE = 10000
ZH_VOCAB_SIZE = 5000
D_MODEL = 256
NHEAD = 8
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
DIM_FEEDFORWARD = 512
DROPOUT = 0.1
BATCH_SIZE = 64
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
SEED = 42
EARLY_STOP_PATIENCE = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    set_seed(SEED)
    print(f'Using device: {device}')

    #构建词表
    print('Building vocabulary...')
    en_sents = []
    zh_sents = []
    with open('../data/ted2020/raw.en', 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= MAX_TRAIN_SAMPLES:
                break
            en_sents.append(line.strip())
    with open('../data/ted2020/raw.zh', 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= MAX_TRAIN_SAMPLES:
                break
            zh_sents.append(line.strip())

    en_vocab = build_vocab(en_sents, EN_VOCAB_SIZE, tokenize_en)
    zh_vocab = build_vocab(zh_sents, ZH_VOCAB_SIZE, tokenize_zh)
    zh_idx2token = {v: k for k, v in zh_vocab.items()}
    print(f'EN vocab size: {len(en_vocab)}, ZH vocab size: {len(zh_vocab)}')

    os.makedirs('../models', exist_ok=True)
    with open('../models/vocab.pkl', 'wb') as f:
        pickle.dump({'en_vocab': en_vocab, 'zh_vocab': zh_vocab}, f)

    #加载数据集
    print('Loading dataset...')
    full_dataset = TranslationDataset(
        '../data/ted2020/raw.en', '../data/ted2020/raw.zh',
        en_vocab, zh_vocab, max_len=MAX_LEN, max_samples=MAX_TRAIN_SAMPLES
    )

    val_size = int(len(full_dataset) * 0.05)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                               collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=collate_fn)
    print(f'Train samples: {len(train_dataset)}, Valid samples: {len(val_dataset)}')

    #初始化模型
    model = TransformerNMT(
        len(en_vocab), len(zh_vocab),
        d_model=D_MODEL, nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT
    ).to(device)

    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    total_steps = NUM_EPOCHS * len(train_loader)
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    #训练
    best_val_loss = float('inf')
    early_stop_count = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0

        for enc_input, dec_input, dec_target in tqdm(train_loader,
                                                      desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [Train]'):
            enc_input = enc_input.to(device)
            dec_input = dec_input.to(device)
            dec_target = dec_target.to(device)

            optimizer.zero_grad()
            output = model(enc_input, dec_input)
            output = output.view(-1, output.size(-1))
            dec_target = dec_target.view(-1)
            loss = criterion(output, dec_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        mean_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for enc_input, dec_input, dec_target in tqdm(val_loader,
                                                          desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [Valid]'):
                enc_input = enc_input.to(device)
                dec_input = dec_input.to(device)
                dec_target = dec_target.to(device)

                output = model(enc_input, dec_input)
                output = output.view(-1, output.size(-1))
                dec_target = dec_target.view(-1)
                loss = criterion(output, dec_target)
                val_loss += loss.item()

        mean_val_loss = val_loss / len(val_loader)

        print(f'Epoch {epoch+1}/{NUM_EPOCHS} | '
              f'Train Loss: {mean_train_loss:.4f} | Valid Loss: {mean_val_loss:.4f}')

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            torch.save(model.state_dict(), '../models/transformer_nmt_best.pth')
            print(f'  -> Best model saved (Val Loss: {best_val_loss:.4f})')
            early_stop_count = 0
        else:
            early_stop_count += 1
            if early_stop_count >= EARLY_STOP_PATIENCE:
                print('Early stopping!')
                break

    print(f'Training completed! Best Val Loss: {best_val_loss:.4f}')
