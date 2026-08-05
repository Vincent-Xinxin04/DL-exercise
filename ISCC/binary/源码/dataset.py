import os
import pandas as pd
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
from preprocess import extract_features, CWE_TO_IDX

class VulnDataset(Dataset):
    def __init__(self, csv_path=None, bin_dir=None, seq_len=4096, is_train=True, cache_path=None, preloaded_data=None):
        self.is_train = is_train
        self.seq_len = seq_len
        self.bin_dir = bin_dir
        
        if preloaded_data is not None:
            self.data = preloaded_data
            self.use_cache = True
        elif cache_path and os.path.exists(cache_path):
            print(f"Loading features from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                self.data = pickle.load(f)
            self.use_cache = True
        else:
            self.df = pd.read_csv(csv_path)
            self.use_cache = False

    def __len__(self):
        return len(self.data) if self.use_cache else len(self.df)

    def __getitem__(self, idx):
        if self.use_cache:
            item = self.data[idx]
            bytes_seq = np.array(item['bytes']).astype(np.int64)
            meta = np.array(item['meta']).astype(np.float32)
            
            bytes_tensor = torch.LongTensor(bytes_seq).view(4096)
            meta_tensor = torch.FloatTensor(meta).view(10)
            
            if self.is_train:
                label = int(item.get('label', 0))
                
                # 安全提取 CWE 标签
                raw_cwe = item.get('cwe_idx', item.get('cwe_id', ""))
                
                # 处理 NaN 或空值
                if pd.isna(raw_cwe) or raw_cwe == "":
                    cwe_idx = -1
                elif isinstance(raw_cwe, str):
                    cwe_idx = CWE_TO_IDX.get(raw_cwe, -1)
                else:
                    cwe_idx = int(raw_cwe)
                
                return bytes_tensor, meta_tensor, torch.tensor(label, dtype=torch.long), torch.tensor(int(cwe_idx), dtype=torch.long)
            
            return bytes_tensor, meta_tensor, item['binary_id']
        else:
            row = self.df.iloc[idx]
            bin_id = row['binary_id']
            file_path = os.path.join(self.bin_dir, f"{bin_id}.exe")
            bytes_seq, meta = extract_features(file_path, self.seq_len)
            
            bytes_tensor = torch.LongTensor(bytes_seq).view(4096)
            meta_tensor = torch.FloatTensor(meta).view(10)
            
            if self.is_train:
                label = int(row['label'])
                raw_cwe = row['cwe_id']
                if pd.isna(raw_cwe) or raw_cwe == "":
                    cwe_idx = -1
                else:
                    cwe_idx = CWE_TO_IDX.get(str(raw_cwe), -1)
                return bytes_tensor, meta_tensor, torch.tensor(label, dtype=torch.long), torch.tensor(int(cwe_idx), dtype=torch.long)
            return bytes_tensor, meta_tensor, bin_id
