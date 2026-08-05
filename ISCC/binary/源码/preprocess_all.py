import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from concurrent.futures import ProcessPoolExecutor
from preprocess import extract_features

# Paths
SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SOURCE_DIR)
DATA_DIR = os.path.join(SOURCE_DIR, 'data')
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
TEST_CSV = os.path.join(DATA_DIR, 'test.csv')
BIN_DIR = os.path.join(DATA_DIR, 'binaries', 'binaries')
MODEL_DIR = os.path.join(BASE_DIR, '模型')

TRAIN_CACHE = os.path.join(MODEL_DIR, 'features_cache.pkl')
TEST_CACHE = os.path.join(MODEL_DIR, 'test_features_cache.pkl')
STAGE2_CACHE = os.path.join(MODEL_DIR, 'stage2_features.pkl')

def process_single_file(args):
    bin_id, label, cwe_id = args
    file_path = os.path.join(BIN_DIR, f"{bin_id}.exe")
    if not os.path.exists(file_path):
        return None
    bytes_seq, meta = extract_features(file_path)
    return {
        'binary_id': bin_id,
        'bytes': np.array(bytes_seq, dtype=np.uint16),
        'meta': np.array(meta, dtype=np.float32),
        'label': label,
        'cwe_idx': cwe_id
    }

def run_preprocessing(csv_path, is_test=False):
    df = pd.read_csv(csv_path)
    tasks = []
    for _, row in df.iterrows():
        if is_test:
            tasks.append((row['binary_id'], 0, ""))
        else:
            tasks.append((row['binary_id'], row['label'], row['cwe_id']))
    
    print(f"Starting preprocessing {len(df)} files for {'TEST' if is_test else 'TRAIN'}...")
    results = []
    with ProcessPoolExecutor() as executor:
        for res in tqdm(executor.map(process_single_file, tasks), total=len(tasks)):
            if res: results.append(res)
    return results

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 1. 提取训练集
    res_train = run_preprocessing(TRAIN_CSV, is_test=False)
    with open(TRAIN_CACHE, 'wb') as f:
        pickle.dump(res_train, f)
    print(f"Saved Train Features to {TRAIN_CACHE}")

    # 2. 提取测试集
    res_test = run_preprocessing(TEST_CSV, is_test=True)
    with open(TEST_CACHE, 'wb') as f:
        pickle.dump(res_test, f)
    print(f"Saved Test Features to {TEST_CACHE}")

    # 3. 合并生成 Stage 2 全量缓存
    print("\nMerging all features for Stage 2...")
    res_stage2 = res_train + res_test
    with open(STAGE2_CACHE, 'wb') as f:
        pickle.dump(res_stage2, f)
    print(f"Saved Stage 2 Features to {STAGE2_CACHE}")
    
    print("\nAll preprocessing done!")

if __name__ == "__main__":
    main()
