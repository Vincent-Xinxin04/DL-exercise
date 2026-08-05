import pickle
import numpy as np
import json
import os

def generate_scaler():
    # 使用相对路径，假设在项目根目录运行
    cache_path = '模型/features_cache.pkl'
    out_path = '模型/feature_scaler.json'
    
    if not os.path.exists(cache_path):
        print(f"Error: {cache_path} not found.")
        return
        
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)

    metas = np.array([d['meta'] for d in data])

    scaler = {
        'min': np.min(metas, axis=0).tolist(),
        'max': np.max(metas, axis=0).tolist(),
        'mean': np.mean(metas, axis=0).tolist()
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(scaler, f, indent=4)

    print(f"Generated {out_path} successfully from training data.")

if __name__ == '__main__':
    generate_scaler()
