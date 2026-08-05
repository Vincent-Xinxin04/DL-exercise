import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import pickle
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
from concurrent.futures import ProcessPoolExecutor

# Add source directory to path
sys.path.append('源码')

from model import BinaryVulnModel
from dataset import VulnDataset
from preprocess import IDX_TO_CWE, extract_features

# Config - Safe relative paths
MODEL_DIR = '模型'
RESULT_DIR = '提交结果'

# 适配 B 榜测试集
TEST_B_BIN_DIR = '源码/test_B/bin_0wbjnTM/bin'
TEST_CACHE = '模型/test_B_features_cache.pkl'
FINAL_MODEL_PATH = '模型/final_model.pth'
OUTPUT_CSV = '提交结果/submission.csv'

BATCH_SIZE = 256

def process_single_file(file_path):
    bin_id = os.path.basename(file_path).replace('.exe', '')
    try:
        bytes_seq, meta = extract_features(file_path)
    except Exception:
        # 异常情况安全回退
        bytes_seq = [256] * 4096
        meta = [0.0] * 10
    return {
        'binary_id': bin_id,
        'bytes': np.array(bytes_seq, dtype=np.uint16),
        'meta': np.array(meta, dtype=np.float32)
    }

def extract_test_features():
    print(f"Scanning for executables in {TEST_B_BIN_DIR}...")
    if not os.path.exists(TEST_B_BIN_DIR):
        print(f"ERROR: Test directory not found at {TEST_B_BIN_DIR}")
        return False
        
    files = [os.path.join(TEST_B_BIN_DIR, f) for f in os.listdir(TEST_B_BIN_DIR) if f.endswith('.exe')]
    print(f"Found {len(files)} executable files.")
    
    if os.path.exists(TEST_CACHE):
        with open(TEST_CACHE, 'rb') as f:
            cached_data = pickle.load(f)
        if len(cached_data) == len(files):
            print(f"Cache {TEST_CACHE} already exists and matches file count, skipping extraction.")
            return True
            
    print("Extracting features using parallel workers...")
    extracted_data = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(executor.map(process_single_file, files), total=len(files), desc="Extracting"))
    
    # Check if we have extracted all files
    for r in results:
        if r is not None:
            extracted_data.append(r)
            
    # Save cache
    os.makedirs(os.path.dirname(TEST_CACHE), exist_ok=True)
    with open(TEST_CACHE, 'wb') as f:
        pickle.dump(extracted_data, f)
    print(f"Extracted {len(extracted_data)} features and saved to {TEST_CACHE}")
    return True

def predict_final():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 确保特征提取
    if not extract_test_features():
        return

    # 2. 准备数据加载器
    print(f"Loading test dataset from {TEST_CACHE}...")
    test_dataset = VulnDataset(is_train=False, cache_path=TEST_CACHE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 3. 加载预测模型
    if not os.path.exists(FINAL_MODEL_PATH):
        print(f"ERROR: Model weights file not found at {FINAL_MODEL_PATH}!")
        return

    print(f"Loading Multi-class Model from {FINAL_MODEL_PATH}...")
    model = BinaryVulnModel(meta_dim=10).to(device)
    model.load_state_dict(torch.load(FINAL_MODEL_PATH, map_location=device))
    model.eval()

    results = []

    # 4. 推理预测 (应用特征对齐以减少域偏移影响)
    print("Loading feature scaler from training data...")
    with open('模型/feature_scaler.json', 'r') as f:
        scaler = json.load(f)
        scaler_min = torch.tensor(scaler['min'], device=device)
        scaler_max = torch.tensor(scaler['max'], device=device)
        scaler_mean = torch.tensor(scaler['mean'], device=device)

    print("Starting inference...")
    with torch.no_grad():
        for bytes_seq, meta, bin_ids in tqdm(test_loader):
            
            # --- 应用训练集特征缩放器 (Standard Feature Scaler) ---
            meta = meta.to(device)
            # 1. 对被混淆或 Strip 导致的缺失特征进行均值填补 (Mean Imputation)
            meta[:, 2] = torch.where(meta[:, 2] == 0.0, scaler_mean[2], meta[:, 2]) # 导入表缺失
            meta[:, 8] = torch.where(meta[:, 8] == 0.0, scaler_mean[8], meta[:, 8]) # 符号表缺失
            
            # 2. 特征裁剪 (Min-Max Scaler Clipping): 限制在训练集特征分布边界内
            meta = torch.clamp(meta, min=scaler_min, max=scaler_max)

            # 3. 对数平滑
            meta = torch.log1p(torch.clamp(meta, min=0))
            bytes_seq = bytes_seq.to(device)
            
            with torch.amp.autocast('cuda'):
                det_out, class_out = model(bytes_seq, meta)
            
            det_probs = F.softmax(det_out, dim=1)
            class_probs = F.softmax(class_out, dim=1)
            
            det_preds = torch.argmax(det_probs, dim=1).cpu().numpy()
            class_preds = torch.argmax(class_probs, dim=1).cpu().numpy()
            
            for i in range(len(bin_ids)):
                label = int(det_preds[i])
                cwe_id = ""
                
                # 仅当模型判定为有漏洞时，输出对应的多分类 CWE 编号
                if label == 1:
                    cwe_id = IDX_TO_CWE[class_preds[i]]
                
                results.append({
                    'binary_id': bin_ids[i],
                    'label': label,
                    'cwe_id': cwe_id
                })

    # 5. 保存结果
    df_res = pd.DataFrame(results)
    df_res.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    print(f"Submission results saved successfully to {OUTPUT_CSV}!")

if __name__ == "__main__":
    predict_final()
