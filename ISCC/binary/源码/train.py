import os
import sys
import subprocess

def run_script(script_name):
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    print(f"\n>>> Running {script_name}...")
    # 使用当前解释器运行子脚本
    result = subprocess.run([sys.executable, script_path], check=True)
    return result

def main():
    print("========== Starting Full Training Pipeline ==========")
    
    try:
        # 1. 基础训练
        run_script("train_stage1.py")
        
        # 2. 构建增强数据集
        run_script("prepare_augmented_data.py")
        
        # 3. 增强微调 (HEM)
        run_script("train_stage2.py")
        
        print("\n[SUCCESS] All training stages completed successfully!")
        print("Final model saved as: 模型/final_model.pth")
        
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Pipeline failed during {e.cmd[1]}")
        sys.exit(1)

if __name__ == "__main__":
    main()
