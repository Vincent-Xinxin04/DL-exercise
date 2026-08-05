# PowerShell 恶意脚本分类项目

本项目是针对 PowerShell 脚本恶意行为检测的分类模型，旨在通过分析脚本的多维特征（如函数作用域、解码活动、网络命令等），实现对脚本类别的精准识别。

## 1. 任务说明
使用机器学习模型判别 PowerShell 脚本属于以下哪类：
- `{0：正常脚本}`
- `{1：一般恶意脚本}`
- `{2：混淆恶意脚本}`

## 2. 项目目录结构
```text
项目根目录/
├─ 源码/                # 模型代码、train + test 脚本
│  ├─ data/             # 数据集 (data_train.csv, data_test.csv)
│  ├─ train.py          # 训练并保存模型
│  └─ test.py           # 加载模型并生成预测结果
├─ 模型/                # 训练好的模型文件 (*.json, *.txt, *.cbm, *.npy, *.json)
├─ 提交结果/            # 最终生成的 submission.csv
├─ docker容器/           # Docker 镜像配置文件 (Dockerfile)
├─ requirements.txt      # Python 环境依赖
└─ README.md             # 项目说明文档
```

## 3. 环境配置

### 本地环境
建议使用 Python 3.10+。安装所需依赖：
```bash
pip install -r requirements.txt
```

### Docker 环境
项目提供了支持 GPU 加速的 Dockerfile。
1. **构建镜像**:
   ```bash
   docker build -t powershell_classifier -f docker容器/Dockerfile .
   ```
2. **运行容器进行预测**:

   **Linux/macOS (Bash)**:
   ```bash
   docker run --rm -v $(pwd)/提交结果:/app/提交结果 powershell_classifier
   ```

   **Windows (PowerShell)**:
   ```powershell
   docker run --rm -v "$PWD/提交结果:/app/提交结果" powershell_classifier
   ```

### 虚拟环境（备选方案）
若无法启动 Docker 环境，可手动创建虚拟环境并安装依赖：

```powershell
# 创建虚拟环境 (Windows)
python -m venv venv

# 创建虚拟环境 (Linux/Mac)
# python3 -m venv venv

# 激活虚拟环境 (Windows)
.\venv\Scripts\activate

# 激活虚拟环境 (Linux/Mac)
# source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 运行预测
python 源码/test.py
```

## 4. 运行方式

### 步骤 1：模型训练
运行以下命令进行模型训练。该脚本会自动执行特征工程、交叉验证、模型集成及参数保存。
```bash
python 源码/train.py
```
*训练后的模型将保存至 `模型/` 文件夹下。*

### 步骤 2：生成预测结果
运行以下命令加载模型并对测试集进行预测，结果将生成在 `提交结果/submission.csv`。
```bash
python 源码/test.py
```

## 5. 模型与特征优化说明

### 核心算法：多模型加权集成 (Ensemble Learning)
- **集成架构**：采用了 **XGBoost**、**LightGBM** 和 **CatBoost** 三大主流梯度提升树模型的加权集成。
- **稳定性保证**：使用了 **3 种随机种子 (3-Seed Averaging)** 以及 **5 折交叉验证 (5-Fold CV)**，总计 45 个子模型进行概率平均，极大地提升了模型的泛化能力和鲁棒性。
- **自适应权重 (Blending)**：通过 OOF (Out-of-Fold) 搜索最优的模型融合权重，平衡各模型的优劣势。

### 特征工程与数据增强
- **高级特征聚合 (Hyper Feature Engineering)**：
  - **行为强度**：统计各类 Profile 的活动频率与多样性。
  - **关键交互**：捕捉“解码活动”与“网络/凭据操作”的交互特征。
  - **混淆深度**：计算布局变化、标识符变化与内容编码的比率，专门针对“混淆恶意脚本”类。
  - **作用域复杂度**：分析函数、分支、循环作用域的分布情况。
- **防泄露目标编码 (Anti-Leakage Target Encoding)**：
  - 针对高基数特征（如行为聚类），采用了严格的 **K-Fold Target Encoding**。
  - 在训练过程中加入了轻微噪声，确保线下验证与线上提交得分的高度一致，有效防止过拟合。

### 类别加权优化 (Class Weighting)
- 针对赛题评估指标 **Macro-F1**，为类别 1（一般恶意）和类别 2（混淆恶意）设置了 **1.2 倍** 的损失权重，提升了对恶意脚本的召回性能。

## 6. 环境要求
- **操作系统**: Windows 10/11 或 Linux (Ubuntu 22.04+)
- **硬件**: 建议支持 CUDA 的 NVIDIA GPU (可加速训练/推理)
- **依赖库**: 详见 `requirements.txt`
