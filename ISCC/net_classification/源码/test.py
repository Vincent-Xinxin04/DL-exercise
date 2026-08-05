import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import os

# =================================================================
# 项目名称：ISCC 2026 网络流量分类
# 版本：V9 提纯版 (Robust Alignment) - 预测脚本
# 功能：自适应特征对齐，解决 108/109 维度微差问题
# =================================================================

CONFIG = {
    'test_path': r'data/test_data_b_5xt7X5r.csv',
    'model_dir': r'../模型/',
    'result_path': r'../提交结果/submission.csv',
    'seeds': [2026, 42, 1024],
    'top_n_interactions': 10
}

OS_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(OS_DIR)

def preprocess_features(df, important_features):
    original_features = [c for c in df.columns if c not in ['id', 'label', 'pattern_mix_density']]
    X = df[original_features].copy()
    
    # 基础统计量
    X['row_mean'] = X[original_features].mean(axis=1)
    X['row_std'] = X[original_features].std(axis=1)
    X['row_max'] = X[original_features].max(axis=1)
    X['row_skew'] = X[original_features].skew(axis=1)
    
    # 10阶交互
    top_feats = important_features[:CONFIG['top_n_interactions']]
    for i, f in enumerate(top_feats):
        X[f'{f}_sq'] = X[f] ** 2
        for f2 in top_feats[i+1:]:
            X[f'{f}_x_{f2}'] = X[f] * X[f2]
    return X

def predict():
    print("正在启动 V9 自适应对齐预测流程...")
    test_df = pd.read_csv(CONFIG['test_path'])
    ids = test_df['id']
    
    le = joblib.load(os.path.join(CONFIG['model_dir'], 'label_encoder.joblib'))
    top_features = joblib.load(os.path.join(CONFIG['model_dir'], 'top_features.joblib'))
    
    # 加载样例模型获取特征需求
    base_m_path = os.path.join(CONFIG['model_dir'], f'lgb_model_fold_0_seed_{CONFIG["seeds"][0]}.joblib')
    model_sample = joblib.load(base_m_path)
    expected_feature_names = model_sample.feature_name()
    print(f"模型预期特征数: {len(expected_feature_names)}")
    
    # 生成全量特征
    X_test_all = preprocess_features(test_df, top_features)
    
    # 核心步骤：强制对齐。只保留模型在训练时见过的特征
    # 如果训练时丢弃了某个常数交互特征，这里也会自动同步丢弃
    X_test_aligned = X_test_all[expected_feature_names]
    print(f"特征对齐完成，输出维度: {X_test_aligned.shape[1]}")
    
    final_probs = np.zeros((len(X_test_aligned), 12))
    model_count = 0
    for seed in CONFIG['seeds']:
        for fold in range(5):
            m_path = os.path.join(CONFIG['model_dir'], f'lgb_model_fold_{fold}_seed_{seed}.joblib')
            if os.path.exists(m_path):
                m = joblib.load(m_path)
                final_probs += m.predict(X_test_aligned)
                model_count += 1
    
    if model_count > 0:
        final_probs /= model_count
        pred_labels = le.inverse_transform(np.argmax(final_probs, axis=1))
        pd.DataFrame({'id': ids, 'label': pred_labels}).to_csv(CONFIG['result_path'], index=False)
        print(f"预测完成！成功集成 {model_count} 个对齐后的 V9 模型。")
    else:
        print("错误：未找到模型文件。")

if __name__ == "__main__":
    predict()
