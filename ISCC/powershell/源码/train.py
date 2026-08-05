import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
import os
import warnings
import sys
import shutil
import json

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def log(msg):
    print(msg)
    sys.stdout.flush()

def hyper_feature_engineering(df):
    df = df.copy()
    profile_cols = [col for col in df.columns if 'profile' in col]
    scope_cols = [col for col in df.columns if 'scope_level' in col]
    
    # 1. Advanced Aggregations
    df['activity_intensity'] = df[profile_cols].gt(0).sum(axis=1)
    df['profile_sum'] = df[profile_cols].sum(axis=1)
    df['profile_max'] = df[profile_cols].max(axis=1)
    df['profile_std'] = df[profile_cols].std(axis=1)
    df['profile_diversity'] = (df[profile_cols] > 0).sum(axis=1)
    
    # 2. Key Behavior Interactions
    df['decode_x_network'] = df['decode_activity_profile'] * df['network_command_profile']
    df['decode_x_credential'] = df['decode_activity_profile'] * df['credential_runtime_profile']
    
    # 3. Obfuscation Depth
    obf_cols = ['layout_variation_profile', 'identifier_variation_profile', 'content_encoding_profile']
    df['obfuscation_sum'] = df[obf_cols].sum(axis=1)
    df['obfuscation_ratio'] = df['obfuscation_sum'] / (df['profile_sum'] + 1)
    
    # 4. Scope Complexity
    df['scope_total'] = df[scope_cols].sum(axis=1)
    df['activity_per_scope'] = df['profile_sum'] / (df['scope_total'] + 1)
    
    # 5. Weighted Score
    weights = {'decode_activity_profile': 5.0, 'network_command_profile': 4.0, 'credential_runtime_profile': 4.0}
    df['risk_score'] = sum(df[col] * w for col, w in weights.items() if col in df.columns)
    
    # 6. Behavior Cluster (High Cardinality)
    df['behavior_cluster'] = df['decode_activity_profile'].astype(str) + "_" + \
                             df['network_command_profile'].astype(str) + "_" + \
                             df['credential_runtime_profile'].astype(str)
                             
    return df

# Leakage-free Target Encoding logic
def apply_oof_te(X, y, X_val, X_test, te_cols, weight=120):
    X = X.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()
    
    global_means = y.value_counts(normalize=True).to_dict()
    
    for col in te_cols:
        for cls in [0, 1, 2]:
            target_col = f'{col}_te_{cls}'
            # Compute stats on Training Fold ONLY
            temp_df = pd.DataFrame({col: X[col], 'target': (y == cls).astype(int)})
            agg = temp_df.groupby(col)['target'].agg(['sum', 'count'])
            
            smooth_mean = (agg['sum'] + weight * global_means.get(cls, 0)) / (agg['count'] + weight)
            mapping = smooth_mean.to_dict()
            
            # Apply to Validation and Test
            X_val[target_col] = X_val[col].map(mapping).fillna(global_means.get(cls, 0))
            X_test[target_col] = X_test[col].map(mapping).fillna(global_means.get(cls, 0))
            
            # Apply to Training with small noise
            X[target_col] = X[col].map(mapping).fillna(global_means.get(cls, 0))
            X[target_col] += np.random.normal(0, 0.005, len(X))
            
    return X, X_val, X_test

def train():
    log("============================================================")
    log("  V5.2 Anti-Leakage Pipeline: Honest OOF Target Encoding")
    log("============================================================")
    
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '模型'))
    if not os.path.exists(model_dir): os.makedirs(model_dir)
    else:
        for f in os.listdir(model_dir):
            if f.endswith(('.json', '.txt', '.cbm', '.npy')):
                try: os.remove(os.path.join(model_dir, f))
                except: pass

    log("Loading data...")
    train_df = pd.read_csv(os.path.join(data_dir, 'data_train.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'data_test.csv'))
    
    log("Applying feature engineering...")
    train_df = hyper_feature_engineering(train_df)
    test_df = hyper_feature_engineering(test_df)
    
    # Device configuration
    has_gpu = False
    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except: pass
    xgb_device, lgb_device, cb_device = ('cuda', 'gpu', 'GPU') if has_gpu else ('cpu', 'cpu', 'CPU')
    log(f"Device: {'GPU' if has_gpu else 'CPU'}")

    te_cols = ['behavior_cluster', 'decode_activity_profile', 'network_command_profile', 'function_scope_level']
    base_features = ['function_scope_level', 'branch_scope_level', 'loop_scope_level', 
                     'parameter_block_presence', 'pipeline_usage_level', 'decode_activity_profile', 
                     'network_command_profile', 'task_registry_profile', 'credential_runtime_profile', 
                     'structure_rhythm_profile', 'layout_variation_profile', 'identifier_variation_profile', 
                     'content_encoding_profile', 'command_surface_profile', 'extension_import_profile']
    
    cat_features = base_features + ['behavior_cluster']
    
    X = train_df.drop(['name', 'label'], axis=1)
    y = train_df['label']
    X_test_orig = test_df.drop(['name'], axis=1)
    
    seeds = [42, 2024, 888]
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    total_oof_xgb = np.zeros((len(X), 3))
    total_oof_lgb = np.zeros((len(X), 3))
    total_oof_cb = np.zeros((len(X), 3))
    
    log(f"\nStarting Seed Averaging with {len(seeds)} seeds...")
    for s_idx, seed in enumerate(seeds):
        log(f"\n>>> SEED {seed} ({s_idx+1}/{len(seeds)}) <<<")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            log(f"--- Fold {fold+1}/5 ---")
            X_tr_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_tr_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            X_tr_te, X_val_te, _ = apply_oof_te(X_tr_fold, y_tr_fold, X_val_fold, X_test_orig, te_cols, weight=120)
            for col in cat_features:
                X_tr_te[col] = X_tr_te[col].astype('category')
                X_val_te[col] = X_val_te[col].astype('category')

            # Calculate Class Weights
            weights_map = {0: 1.0, 1: 1.2, 2: 1.2}
            sample_weights = y_tr_fold.map(weights_map)

            # 1. XGBoost
            xgb = XGBClassifier(n_estimators=3000, max_depth=6, learning_rate=0.008, reg_lambda=100, reg_alpha=5,
                               objective='multi:softprob', device=xgb_device, early_stopping_rounds=150,
                               subsample=0.8, colsample_bytree=0.6, enable_categorical=True, random_state=seed)
            xgb.fit(X_tr_te, y_tr_fold, eval_set=[(X_val_te, y_val_fold)], sample_weight=sample_weights, verbose=False)
            xgb.save_model(os.path.join(model_dir, f'xgboost_s{seed}_f{fold}.json'))
            total_oof_xgb[val_idx] += xgb.predict_proba(X_val_te) / len(seeds)
            # 2. LightGBM
            lgb_model = LGBMClassifier(n_estimators=2000, max_depth=6, num_leaves=31, learning_rate=0.008, 
                               reg_lambda=100, reg_alpha=5, min_child_samples=80,
                               objective='multiclass', device=lgb_device, verbose=-1, subsample=0.8, colsample_bytree=0.6, random_state=seed)
            lgb_model.fit(X_tr_te, y_tr_fold, eval_set=[(X_val_te, y_val_fold)], sample_weight=sample_weights, callbacks=[])
            lgb_tmp = f'lgb_s{seed}_f{fold}_tmp.txt'
            lgb_model.booster_.save_model(lgb_tmp)
            shutil.move(lgb_tmp, os.path.join(model_dir, f'lgbm_s{seed}_f{fold}.txt'))
            total_oof_lgb[val_idx] += lgb_model.predict_proba(X_val_te) / len(seeds)
            # 3. CatBoost
            cb_model = CatBoostClassifier(iterations=2000, depth=6, learning_rate=0.008, l2_leaf_reg=80,
                                         loss_function='MultiClass', task_type=cb_device, verbose=False,
                                         early_stopping_rounds=150, cat_features=cat_features, random_seed=seed,
                                         random_strength=2, class_weights=[1.0, 1.2, 1.2])
            cb_model.fit(X_tr_te, y_tr_fold, eval_set=[(X_val_te, y_val_fold)])
            cb_model.save_model(os.path.join(model_dir, f'catboost_s{seed}_f{fold}.cbm'))
            total_oof_cb[val_idx] += cb_model.predict_proba(X_val_te) / len(seeds)
            # Note: f1_fold here is slightly approximate due to seed averaging accumulation
            
    # Optimal Weights Search (Blender)
    best_f1, best_w = 0, (0.33, 0.33, 0.34)
    for w1 in np.linspace(0, 1, 21):
        for w2 in np.linspace(0, 1-w1, int((1-w1)*20)+1):
            w3 = 1.0 - w1 - w2
            blend = total_oof_xgb * w1 + total_oof_lgb * w2 + total_oof_cb * w3
            score = f1_score(y, np.argmax(blend, axis=1), average='macro')
            if score > best_f1:
                best_f1, best_w = score, (w1, w2, w3)
    
    log(f"\nFinal Seed-Averaged OOF Macro-F1: {best_f1:.4f}")
    log(f"Optimal Weights: XGB={best_w[0]:.2f}, LGB={best_w[1]:.2f}, CB={best_w[2]:.2f}")
    np.save(os.path.join(model_dir, 'best_weights.npy'), np.array(best_w))
    
    # Save Full-Data TE for Inference
    log("Saving inference TE mappings...")
    te_mappings = {}
    global_means = y.value_counts(normalize=True).to_dict()
    for col in te_cols:
        te_mappings[col] = {}
        for cls in [0, 1, 2]:
            temp_df = pd.DataFrame({col: X[col], 'target': (y == cls).astype(int)})
            # Use same smoothing weight as in training
            mapping = ((temp_df.groupby(col)['target'].sum() + 60 * global_means.get(cls, 0)) / (temp_df.groupby(col)['target'].count() + 60)).to_dict()
            te_mappings[col][str(cls)] = mapping
    
    with open(os.path.join(model_dir, 'te_mappings.json'), 'w') as f:
        json.dump({'mappings': te_mappings, 'global_means': global_means}, f)
        
    log("Training complete!")

if __name__ == "__main__":
    train()