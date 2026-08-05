import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
import os
import warnings
import shutil
import tempfile
import json

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def hyper_feature_engineering(df):
    df = df.copy()
    profile_cols = [col for col in df.columns if 'profile' in col]
    scope_cols = [col for col in df.columns if 'scope_level' in col]
    
    df['activity_intensity'] = df[profile_cols].gt(0).sum(axis=1)
    df['profile_sum'] = df[profile_cols].sum(axis=1)
    df['profile_max'] = df[profile_cols].max(axis=1)
    df['profile_std'] = df[profile_cols].std(axis=1)
    df['profile_diversity'] = (df[profile_cols] > 0).sum(axis=1)
    
    df['decode_x_network'] = df['decode_activity_profile'] * df['network_command_profile']
    df['decode_x_credential'] = df['decode_activity_profile'] * df['credential_runtime_profile']
    
    obf_cols = ['layout_variation_profile', 'identifier_variation_profile', 'content_encoding_profile']
    df['obfuscation_sum'] = df[obf_cols].sum(axis=1)
    df['obfuscation_ratio'] = df['obfuscation_sum'] / (df['profile_sum'] + 1)
    
    df['scope_total'] = df[scope_cols].sum(axis=1)
    df['activity_per_scope'] = df['profile_sum'] / (df['scope_total'] + 1)
    
    weights = {'decode_activity_profile': 5.0, 'network_command_profile': 4.0, 'credential_runtime_profile': 4.0}
    df['risk_score'] = sum(df[col] * w for col, w in weights.items() if col in df.columns)
    df['behavior_cluster'] = df['decode_activity_profile'].astype(str) + "_" + \
                             df['network_command_profile'].astype(str) + "_" + \
                             df['credential_runtime_profile'].astype(str)
    return df

def test():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '模型'))
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '提交结果'))
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    test_df = pd.read_csv(os.path.join(data_dir, 'data_test_B.csv'))
    
    print("Applying hyper-feature engineering...")
    test_df = hyper_feature_engineering(test_df)
    
    # Load TE mappings
    te_path = os.path.join(model_dir, 'te_mappings.json')
    if os.path.exists(te_path):
        with open(te_path, 'r') as f:
            te_data = json.load(f)
            mappings = te_data['mappings']
            global_means = te_data['global_means']
            for col, cls_maps in mappings.items():
                for cls, mapping in cls_maps.items():
                    test_df[f'{col}_te_{cls}'] = test_df[col].astype(str).map(mapping).fillna(global_means.get(str(cls), 0.33))
    
    base_features = ['function_scope_level', 'branch_scope_level', 'loop_scope_level', 
                     'parameter_block_presence', 'pipeline_usage_level', 'decode_activity_profile', 
                     'network_command_profile', 'task_registry_profile', 'credential_runtime_profile', 
                     'structure_rhythm_profile', 'layout_variation_profile', 'identifier_variation_profile', 
                     'content_encoding_profile', 'command_surface_profile', 'extension_import_profile']
    
    cat_features = base_features + ['behavior_cluster']
    
    # Load training dataset to align categories and prevent XGBoost/CatBoost unseen category errors
    train_path = os.path.join(data_dir, 'data_train.csv')
    if os.path.exists(train_path):
        print("Loading training dataset to align feature categories...")
        train_df = pd.read_csv(train_path)
        train_df = hyper_feature_engineering(train_df)
        for col in cat_features:
            train_categories = list(train_df[col].unique())
            mode_value = train_df[col].mode()[0]
            
            # Map any unseen values in the test set to the training set's mode (most frequent value)
            is_unseen = ~test_df[col].isin(train_categories)
            if is_unseen.any():
                test_df.loc[is_unseen, col] = mode_value
                
            # Convert to Categorical matching the training set's categories exactly (prevents NaN values)
            test_df[col] = pd.Categorical(test_df[col], categories=train_categories)
            
    X_test = test_df.drop(['name'], axis=1)
    for col in cat_features: X_test[col] = X_test[col].astype('category')
    
    seeds = [42, 2024, 888]
    n_splits = 5
    weights_path = os.path.join(model_dir, 'best_weights.npy')
    weights = np.load(weights_path) if os.path.exists(weights_path) else np.array([0.33, 0.33, 0.34])
    
    all_probs = np.zeros((len(X_test), 3))
    print(f"Loading Ensemble (3 Seeds x 5 Folds)...")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        for seed in seeds:
            for fold in range(n_splits):
                # XGBoost
                xgb_path = os.path.join(model_dir, f'xgboost_s{seed}_f{fold}.json')
                if os.path.exists(xgb_path):
                    model = XGBClassifier(); model.load_model(xgb_path)
                    all_probs += model.predict_proba(X_test) * weights[0] / (len(seeds) * n_splits)
                # LightGBM
                lgb_path = os.path.join(model_dir, f'lgbm_s{seed}_f{fold}.txt')
                if os.path.exists(lgb_path):
                    tmp_lgb = os.path.join(tmp_dir, f"tmp_lgb_{seed}_{fold}.txt"); shutil.copy(lgb_path, tmp_lgb)
                    model = lgb.Booster(model_file=tmp_lgb)
                    all_probs += model.predict(X_test) * weights[1] / (len(seeds) * n_splits)
                # CatBoost
                cb_path = os.path.join(model_dir, f'catboost_s{seed}_f{fold}.cbm')
                if os.path.exists(cb_path):
                    model = CatBoostClassifier(); model.load_model(cb_path)
                    all_probs += model.predict_proba(X_test) * weights[2] / (len(seeds) * n_splits)
    
    final_preds = np.argmax(all_probs, axis=1)
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '提交结果'))
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    pd.DataFrame({'name': test_df['name'], 'label': final_preds.astype(int)}).to_csv(os.path.join(output_dir, 'submission.csv'), index=False)
    print("Done.")

if __name__ == "__main__":
    test()
