import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import re
from tqdm import tqdm
import os
# 从 train.py 导入基础配置和类
from train import LogDataset, LogModel, clean_log, MAX_LINES, MAX_LEN, TYPE_TO_IDX, IDX_TO_TYPE, NUM_FOLDS

# 修复 Windows 环境下的 OpenMP 报错
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 路径配置（统一使用相对路径）
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(SRC_DIR, 'data')
TEST_CSV_B = os.path.join(DATA_DIR, 'test_data_b.csv')
TEST_CSV_A = os.path.join(DATA_DIR, 'test.csv')
TEST_CSV = TEST_CSV_B if os.path.exists(TEST_CSV_B) else TEST_CSV_A
MODEL_DIR = os.path.join(BASE_DIR, '模型')
VOCAB_PATH = os.path.join(MODEL_DIR, 'vocab.pth')
RESULT_DIR = os.path.join(BASE_DIR, '提交结果')
SUBMISSION_CSV = os.path.join(RESULT_DIR, 'submission.csv')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def inference():
    print(f"正在从 {VOCAB_PATH} 加载词表...")
    if not os.path.exists(VOCAB_PATH):
        print(f"错误: 未找到词表文件！")
        return
    vocab = torch.load(VOCAB_PATH)

    print(f"正在加载 {NUM_FOLDS} 个交叉验证模型...")
    models = []
    checkpoint_vocab_size = None
    for fold in range(NUM_FOLDS):
        model_path = os.path.join(MODEL_DIR, f'model_fold_{fold}_final.pth')
        if not os.path.exists(model_path):
            model_path = os.path.join(MODEL_DIR, f'model_fold_{fold}.pth')
            
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=DEVICE)
            if checkpoint_vocab_size is None:
                checkpoint_vocab_size = state_dict['embedding.weight'].shape[0]
            
            model = LogModel(checkpoint_vocab_size).to(DEVICE)
            model.load_state_dict(state_dict)
            model.eval()
            models.append(model)
        else:
            print(f"警告: 未找到第 {fold} 折的模型文件。")
    
    if not models:
        print("错误: 未找到任何有效的模型权重！")
        return

    print("正在加载测试数据...")
    df_test = pd.read_csv(TEST_CSV)
    test_dataset = LogDataset(df_test, vocab=vocab, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    results = []
    
    with torch.no_grad():
        for i, x in enumerate(tqdm(test_loader, desc="执行推理")):
            x = x.to(DEVICE)
            # 模型集成：平均所有模型的 Logits 概率
            fold_probs = []
            for model in models:
                logits = model(x)
                probs = torch.softmax(logits, dim=-1)
                fold_probs.append(probs)
            
            avg_probs = torch.stack(fold_probs).mean(0).cpu().numpy()
            
            for j in range(avg_probs.shape[0]):
                sample_idx = len(results)
                if sample_idx >= len(df_test): break
                
                probs = avg_probs[j]
                # 判定阈值设为 0.2 (根据 OOF 搜索得到的最优参数，能将 F1_detect 从 0.81 提升至 0.91)
                is_anomaly = probs[:, 0] < 0.2
                sample_preds = np.zeros(MAX_LINES, dtype=int)
                sample_preds[is_anomaly] = np.argmax(probs[is_anomaly, 1:], axis=1) + 1
                
                # 验证过的最佳策略：空隙填充 (GAP=3)
                max_gap = 3
                for gap in range(1, max_gap + 1):
                    for k in range(1, len(sample_preds) - gap):
                        if sample_preds[k:k+gap].sum() == 0: 
                            if sample_preds[k-1] != 0 and sample_preds[k-1] == sample_preds[k+gap]:
                                sample_preds[k:k+gap] = sample_preds[k-1]
                        
                # 重建异常区间
                spans = []
                current_span = None
                for k, p in enumerate(sample_preds):
                    if p > 0:
                        atype = IDX_TO_TYPE[p]
                        if current_span and current_span['type'] == atype:
                            current_span['end'] = k
                        else:
                            if current_span: spans.append(current_span)
                            current_span = {'start': k, 'end': k, 'type': atype}
                    else:
                        if current_span: spans.append(current_span); current_span = None
                if current_span: spans.append(current_span)
                
                # 后处理过滤逻辑 (MIN_LEN=2)
                spans = LogModel.post_process_spans(spans)
                
                # 构造提交格式
                has_anomaly = 1 if spans else 0
                if has_anomaly:
                    primary = spans[0] # 本地验证显示，主异常取第一个效果最好
                    p_start, p_end, p_type = primary['start'], primary['end'], primary['type']
                    all_spans_str = ';'.join([f"{s['start']}|{s['end']}|{s['type']}" for s in spans])
                else:
                    p_start, p_end, p_type, all_spans_str = -1, -1, 'none', ''
                
                results.append({
                    'id': df_test.iloc[sample_idx]['id'],
                    'has_anomaly': has_anomaly,
                    'primary_start_idx': p_start,
                    'primary_end_idx': p_end,
                    'primary_anomaly_type': p_type,
                    'all_spans': all_spans_str
                })

    print(f"正在保存结果至 {SUBMISSION_CSV}...")
    os.makedirs(RESULT_DIR, exist_ok=True)
    pd.DataFrame(results).to_csv(SUBMISSION_CSV, index=False)
    print("完成。")

if __name__ == "__main__":
    inference()
