import pandas as pd
import os

# 配置路径
SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SOURCE_DIR)
TRAIN_CSV = os.path.join(SOURCE_DIR, 'data', 'train.csv')
SUBMISSION_CSV = os.path.join(BASE_DIR, '提交结果', 'submission.csv')
OUTPUT_CSV = os.path.join(SOURCE_DIR, 'data', 'augmented_train.csv')

def prepare_data():
    print("正在准备增强数据集 (Pseudo-labeling)...")
    
    # 1. 加载原始训练集
    if not os.path.exists(TRAIN_CSV):
        print(f"错误: 找不到原始训练集 {TRAIN_CSV}")
        return
    df_train = pd.read_csv(TRAIN_CSV)
    print(f"原始训练集样本数: {len(df_train)}")
    
    # 2. 加载预测结果作为伪标签
    if not os.path.exists(SUBMISSION_CSV):
        print(f"提示: 找不到预测文件 {SUBMISSION_CSV}。")
        print("如果是首次运行，请先通过 Stage 1 模型生成预测结果。")
        return
        
    df_pseudo = pd.read_csv(SUBMISSION_CSV)
    print(f"引入伪标签样本数: {len(df_pseudo)}")
    
    # 3. 合并数据
    # 将基础训练集与预测出的伪标签数据合并，用于第二阶段模型精炼
    df_augmented = pd.concat([df_train, df_pseudo], ignore_index=True)
    
    # 4. 保存结果
    df_augmented.to_csv(OUTPUT_CSV, index=False)
    print(f"增强数据集已保存至: {OUTPUT_CSV}")
    print(f"总样本数: {len(df_augmented)}")

if __name__ == "__main__":
    prepare_data()
