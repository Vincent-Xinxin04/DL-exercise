import pandas as pd
import os

def create_pseudo_dataset():
    src_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(src_dir, 'data')
    res_dir = os.path.join(os.path.dirname(src_dir), '提交结果')
    
    train_csv = os.path.join(data_dir, 'train.csv')
    test_csv = os.path.join(data_dir, 'test.csv')
    sub_csv = os.path.join(res_dir, 'submission.csv')
    pseudo_csv = os.path.join(data_dir, 'train_pseudo.csv')
    
    print(f"加载原训练集: {train_csv}")
    df_train = pd.read_csv(train_csv)
    
    print(f"加载测试集日志: {test_csv}")
    df_test = pd.read_csv(test_csv)
    
    print(f"加载测试集伪标签: {sub_csv}")
    df_sub = pd.read_csv(sub_csv)
    
    # 确保排序一致，方便合并
    df_test = df_test.sort_values('id').reset_index(drop=True)
    df_sub = df_sub.sort_values('id').reset_index(drop=True)
    
    # 检查字段是否匹配
    assert len(df_test) == len(df_sub), "测试集和预测结果数量不一致！"
    
    # 合并测试集特征和预测标签
    df_pseudo = pd.merge(df_test, df_sub, on='id')
    
    # 拼接训练集和伪标签测试集
    df_combined = pd.concat([df_train, df_pseudo], ignore_index=True)
    
    print(f"\n--- 数据集合并报告 ---")
    print(f"原训练集样本数: {len(df_train)}")
    print(f"测试集伪标签数: {len(df_pseudo)}")
    print(f"合并后总样本数: {len(df_combined)}")
    
    df_combined.to_csv(pseudo_csv, index=False)
    print(f"包含伪标签的新数据集已保存至: {pseudo_csv}")

if __name__ == "__main__":
    create_pseudo_dataset()
