import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from tqdm import tqdm

# =================================================================
# 项目名称：ISCC 2026 网络流量分类
# 版本：V9 提纯版 (Refined Generalization)
# 核心特性：特征提纯(移除_bin)、标签平滑加深(0.1)、高鲁棒特征抽样(0.5)
# =================================================================

CONFIG = {
    'train_path': r'data/train_data.csv',
    'test_path': r'data/test_data.csv',
    'model_dir': r'../模型/',
    'top_n_interactions': 10,
    'seeds': [2026, 42, 1024],
    'lgb_params': {
        'objective': 'multiclass',
        'num_class': 12,
        'metric': 'multi_logloss',
        'label_smoothing': 0.1,    # 加深平滑
        'verbosity': -1,
        'learning_rate': 0.02,
        'num_leaves': 31,
        'feature_fraction': 0.5,   # 降低特征比例，提升集成鲁棒性
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 50,
        'device': 'gpu',
        'n_jobs': -1
    }
}

OS_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(OS_DIR)

def preprocess_features(df, important_features=None):
    original_features = [c for c in df.columns if c not in ['id', 'label', 'pattern_mix_density']]
    X = df[original_features].copy()
    X['row_mean'] = X[original_features].mean(axis=1)
    X['row_std'] = X[original_features].std(axis=1)
    X['row_max'] = X[original_features].max(axis=1)
    X['row_skew'] = X[original_features].skew(axis=1)
    
    if important_features is not None:
        top_feats = important_features[:CONFIG['top_n_interactions']]
        for i, f in enumerate(top_feats):
            X[f'{f}_sq'] = X[f] ** 2
            # V9 移除手动分桶 _bin 特征，减少噪音
            for f2 in top_feats[i+1:]:
                X[f'{f}_x_{f2}'] = X[f] * X[f2]
    return X

def add_noise_to_subset(X, scale=0.05):
    X_noise = X.copy()
    # 既然移除了 _bin，现在对全量特征进行加噪
    for col in X.columns:
        std = X[col].std()
        if std > 0:
            X_noise[col] += np.random.normal(0, std * scale, size=len(X))
    return X_noise

def train():
    print("正在启动 V9 提纯版：特征去噪与深度泛化方案...")
    train_df = pd.read_csv(CONFIG['train_path'])
    test_df = pd.read_csv(CONFIG['test_path'])
    
    X_init = preprocess_features(train_df)
    le = LabelEncoder()
    y = le.fit_transform(train_df['label'])
    joblib.dump(le, os.path.join(CONFIG['model_dir'], 'label_encoder.joblib'))
    
    init_model = lgb.LGBMClassifier(n_estimators=100, device='gpu', random_state=42)
    init_model.fit(X_init, y)
    top_features = pd.DataFrame({'f': X_init.columns, 'i': init_model.feature_importances_}).sort_values('i', ascending=False)['f'].tolist()
    joblib.dump(top_features, os.path.join(CONFIG['model_dir'], 'top_features.joblib'))
    
    X = preprocess_features(train_df, top_features)
    X_test = preprocess_features(test_df, top_features)
    
    X_adv = pd.concat([X, X_test], axis=0).reset_index(drop=True)
    y_adv = np.array([0]*len(X) + [1]*len(X_test))
    adv_model = lgb.LGBMClassifier(n_estimators=100, device='gpu', random_state=42)
    adv_model.fit(X_adv, y_adv)
    train_probs = np.clip(adv_model.predict_proba(X)[:,1], 0.01, 0.99)
    # 沿用 V6.2 最佳幂次 0.4
    weights = np.power(train_probs/(1-train_probs), 0.4)
    weights /= weights.mean()
    
    for seed in CONFIG['seeds']:
        print(f"\n--- 种子 {seed} 训练开始 (V9 提纯版) ---")
        np.random.seed(seed)
        
        test_probs = np.zeros((len(X_test), 12))
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, y_tr, w_tr = X.iloc[tr_idx], y[tr_idx], weights[tr_idx]
            X_tr_noise = add_noise_to_subset(X_tr, scale=0.05)
            X_tr_aug = pd.concat([X_tr, X_tr_noise], axis=0)
            y_tr_aug = np.concatenate([y_tr, y_tr])
            w_tr_aug = np.concatenate([w_tr, w_tr])
            dtrain = lgb.Dataset(X_tr_aug, label=y_tr_aug, weight=w_tr_aug)
            model = lgb.train(CONFIG['lgb_params'], dtrain, num_boost_round=800)
            test_probs += model.predict(X_test) / 5
            
        test_preds = np.argmax(test_probs, axis=1)
        conf = np.max(test_probs, axis=1)
        mask = []
        for i in range(len(test_preds)):
            p_class = test_preds[i]
            th = 0.95 if p_class == 3 else (0.85 if p_class in [5,6,7,8,9] else 0.90)
            mask.append(conf[i] > th)
        mask = np.array(mask)
        X_ps, y_ps = X_test[mask], test_preds[mask]
        w_ps = 1.0 + np.power(conf[mask], 2)
        
        X_final_orig = pd.concat([X, X_ps], axis=0).reset_index(drop=True)
        y_final_orig = np.concatenate([y, y_ps])
        w_final_orig = np.concatenate([weights, w_ps])
        
        skf_final = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (tr_idx, va_idx) in enumerate(tqdm(skf_final.split(X_final_orig, y_final_orig), total=5, desc=f"Seed {seed}")):
            X_tr_f, y_tr_f, w_tr_f = X_final_orig.iloc[tr_idx], y_final_orig[tr_idx], w_final_orig[tr_idx]
            X_tr_f_noise = add_noise_to_subset(X_tr_f, scale=0.05)
            X_tr_f_aug = pd.concat([X_tr_f, X_tr_f_noise], axis=0)
            y_tr_f_aug = np.concatenate([y_tr_f, y_tr_f])
            w_tr_f_aug = np.concatenate([w_tr_f, w_tr_f])
            
            dtrain = lgb.Dataset(X_tr_f_aug, label=y_tr_f_aug, weight=w_tr_f_aug)
            dval = lgb.Dataset(X_final_orig.iloc[va_idx], label=y_final_orig[va_idx], reference=dtrain)
            model = lgb.train(
                CONFIG['lgb_params'], dtrain, valid_sets=[dval], valid_names=['valid'],
                num_boost_round=2500, callbacks=[lgb.early_stopping(100), lgb.log_evaluation(period=0)]
            )
            joblib.dump(model, os.path.join(CONFIG['model_dir'], f'lgb_model_fold_{fold}_seed_{seed}.joblib'))

if __name__ == "__main__":
    train()
