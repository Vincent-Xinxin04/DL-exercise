import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

# 路径配置（统一使用相对路径）
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SRC_DIR, 'data')
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')

def run_eda():
    print(f"正在加载数据: {TRAIN_CSV}")
    if not os.path.exists(TRAIN_CSV):
        print("错误: 未找到训练集文件！")
        return
        
    df = pd.read_csv(TRAIN_CSV)
    
    print("\n--- 基础统计 ---")
    print(f"总样本数: {len(df)}")
    print(f"异常样本比例: {df['has_anomaly'].mean():.2%}")
    
    print("\n--- 异常类型分布 ---")
    # 统计所有异常区间的类型
    all_types = []
    for spans in df[df['has_anomaly']==1]['all_spans']:
        for s in str(spans).split(';'):
            try:
                all_types.append(s.split('|')[2])
            except: continue
    
    type_counts = Counter(all_types)
    for t, c in type_counts.most_common():
        print(f"{t:25}: {c}")

    # 绘制异常分布图
    plt.figure(figsize=(12, 6))
    sns.countplot(y=all_types, order=[t[0] for t in type_counts.most_common()])
    plt.title("异常类型分布直方图")
    plt.xlabel("出现次数")
    plt.show()

    print("\n--- 日志长度分析 ---")
    df['line_count'] = df['log_text'].apply(lambda x: len(str(x).split('\n')))
    print(f"平均行数: {df['line_count'].mean():.1f}")
    print(f"最大行数: {df['line_count'].max()}")
    print(f"最小行数: {df['line_count'].min()}")

if __name__ == "__main__":
    run_eda()
