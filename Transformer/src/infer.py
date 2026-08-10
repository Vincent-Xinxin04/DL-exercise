from train import (
    TransformerNMT, tokenize_en, encode, PAD_IDX, SOS_IDX, EOS_IDX,
    UNK_IDX, UNK_TOKEN, device
)
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import os
from tqdm import tqdm

# ==================== 超参数 ====================
MAX_LEN = 50
D_MODEL = 256
NHEAD = 8
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
DIM_FEEDFORWARD = 512
DROPOUT = 0.1

class TestDataset(Dataset):
    def __init__(self, en_path, en_vocab, max_len=50):
        self.max_len = max_len
        self.en_vocab = en_vocab
        self.data = []

        with open(en_path, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = tokenize_en(line)
                if len(tokens) > 0:
                    ids = [SOS_IDX] + encode(tokens, en_vocab)[:max_len] + [EOS_IDX]
                    self.data.append(torch.tensor(ids))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    return pad_sequence(batch, batch_first=True, padding_value=PAD_IDX)

if __name__ == '__main__':
    print(f'Using device: {device}')

    #加载词表
    with open('../models/vocab.pkl', 'rb') as f:
        vocab_data = pickle.load(f)
    en_vocab = vocab_data['en_vocab']
    zh_vocab = vocab_data['zh_vocab']
    zh_idx2token = {v: k for k, v in zh_vocab.items()}

    #初始化模型
    model = TransformerNMT(
        len(en_vocab), len(zh_vocab),
        d_model=D_MODEL, nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT
    ).to(device)

    model_path = '../models/transformer_nmt_best.pth'
    if not os.path.exists(model_path):
        print(f'Model not found: {model_path}')
        print('Please run train.py first to train the model.')
        exit(1)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print('Model loaded successfully!')

    #测试集批量推理
    print('Running inference on test set...')
    test_dataset = TestDataset('../data/test/test.en', en_vocab, max_len=MAX_LEN)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                              collate_fn=collate_fn)

    results = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch = batch.to(device)
            output_batch = model.greedy_decode(batch, MAX_LEN)
            for ids in output_batch.cpu().tolist():
                tokens = []
                for idx in ids:
                    if idx in (SOS_IDX, EOS_IDX, PAD_IDX):
                        continue
                    tokens.append(zh_idx2token.get(idx, UNK_TOKEN))
                results.append(''.join(tokens))

    with open('../data/submission.zh', 'w', encoding='utf-8') as f:
        for res in results:
            f.write(res + '\n')
    print(f'Saved {len(results)} translations to ../data/submission.zh')
