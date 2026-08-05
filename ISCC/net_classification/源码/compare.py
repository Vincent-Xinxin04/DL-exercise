import pandas as pd
import numpy as np
import sys
import os

def compare_submissions(file1, file2):
    if not os.path.exists(file1) or not os.path.exists(file2):
        print(f"错误：找不到文件 {file1} 或 {file2}")
        return

    df1 = pd.read_csv(file1).sort_values('id').reset_index(drop=True)
    df2 = pd.read_csv(file2).sort_values('id').reset_index(drop=True)

    if len(df1) != len(df2):
        print("警告：两个文件的样本数量不一致！")
    
    # 1. 计算一致性
    overlap = (df1['label'] == df2['label']).mean()
    diff_count = (df1['label'] != df2['label']).sum()
    
    print("="*30)
    print(f"对比结果: {os.path.basename(file1)} vs {os.path.basename(file2)}")
    print(f"一致性 (Agreement): {overlap:.4%}")
    print(f"差异样本数 (Differences): {diff_count} / {len(df1)}")
    print("="*30)

    # 2. 类别分布对比
    print("\n类别分布对比 (Distribution):")
    dist1 = df1['label'].value_counts(normalize=True).sort_index()
    dist2 = df2['label'].value_counts(normalize=True).sort_index()
    
    dist_df = pd.DataFrame({
        'Class': dist1.index,
        f'{os.path.basename(file1)} (%)': (dist1.values * 100).round(2),
        f'{os.path.basename(file2)} (%)': (dist2.values * 100).round(2),
        'Diff (%)': ((dist1.values - dist2.values) * 100).round(2)
    })
    print(dist_df.to_string(index=False))

if __name__ == "__main__":
    # 默认对比 提交结果/ 下的两个文件
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    f1 = os.path.join(base_dir, '提交结果', 'submission.csv')
    f2 = os.path.join(base_dir, '提交结果', 'submission_best.csv')
    
    # 如果通过命令行传入参数，则使用参数
    if len(sys.argv) == 3:
        f1 = sys.argv[1]
        f2 = sys.argv[2]
    
    compare_submissions(f1, f2)
