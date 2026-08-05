# 🏆 ISCC 2026 网络流量安全事件类别判别模型

本项目为 ISCC 2026 网络安全事件类别判别赛题的参赛方案。模型基于 **LightGBM** 算法，通过 **V9 提纯版 (Refined Generalization)** 策略，实现了针对强漂移环境下网络流量的高性能、高鲁棒性分类。

---

## 📂 项目目录结构

根据赛题要求，本项目目录结构如下：

```text
项目根目录/
├── 源码/                # 所有模型代码、train + test 脚本
│   ├── train.py         # 训练并保存模型脚本
│   ├── test.py          # 加载模型并生成预测结果脚本
│   └── data/            # 存放训练集(train_data.csv)与测试集(test_data.csv)
├── 模型/                # 训练好的模型文件 (*.joblib)
├── 提交结果/            # 比赛提交的最终预测结果 (submission.csv)
├── docker容器/           # Docker 镜像配置文件 (Dockerfile)
├── requirements.txt      # 环境依赖
└── README.md             # 本说明文件
```

---

## 🛠️ 环境配置

### 1. 本地环境
建议使用 Python 3.13 环境。安装依赖：
```bash
pip install -r requirements.txt
```

### 2. Docker 环境
项目提供了基于 **Ubuntu 24.04 + CUDA 12.6 + Python 3.12** 的容器环境，可直接构建镜像以完美复现：

```powershell
# 1. 在项目根目录下执行构建
docker build -t net-classification -f docker容器/Dockerfile .

# 2. 运行预测
# 使用 -v 参数将本地的“提交结果”文件夹映射进容器，确保结果能保存到本地硬盘
# Windows PowerShell 环境：
docker run --rm -v "${PWD}/提交结果:/app/提交结果" net-classification

# Linux/Bash 环境：
# docker run --rm -v "$(pwd)/提交结果:/app/提交结果" net-classification

# 如需启用 GPU 支持 (需安装 NVIDIA Container Toolkit)：
# docker run --rm --gpus all -v "${PWD}/提交结果:/app/提交结果" net-classification
```

### 3. 虚拟环境（备选方案）
若无法启动 Docker 环境，可手动创建虚拟环境进行替代：

**创建虚拟环境：**
```powershell
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境 (Windows)
.\venv\Scripts\activate

# 激活虚拟环境 (Linux/Mac)
# source venv/bin/activate
```

**安装依赖：**
```bash
pip install -r requirements.txt
```

**验证环境并运行：**
```bash
python 源码/test.py
```

---

## 🚀 运行方式

### 1. 模型训练
运行 `源码/train.py` 将自动完成特征工程、对抗权重校准、模型集成训练并保存至 `模型/` 目录。
```bash
python 源码/train.py
```

### 2. 生成预测结果
运行 `源码/test.py` 将加载 `模型/` 中的集成模型，对 `data/test_data.csv` 进行预测，并输出结果至 `提交结果/submission.csv`。
```bash
python 源码/test.py
```

---

## 🧠 核心优化策略 (V9 提纯版)

本方案针对赛题数据的特征分布与噪声特性，实施了以下核心优化：

### 1. 特征工程与提纯 (Feature Pruning)
*   **移除冗余分桶**：移除了手动生成的 `_bin` 分桶特征。实验证明，在数据分布存在显著漂移时，分桶特征易引入阶梯状噪声，移除后模型决策边界更加平滑，泛化性能显著提升。
*   **高阶交互特征**：保留了 Top-10 重要特征的平方项与两两乘积项，捕捉流量行为间的非线性关系。

### 2. 数据增强与噪声平滑 (Data Augmentation)
*   **高斯噪声注入**：在训练过程中，对特征注入特定比例（Scale=0.05）的高斯噪声，强制模型学习更稳健的特征表示，防止过拟合。
*   **标签平滑 (Label Smoothing)**：设置 `label_smoothing=0.1`，增强模型对训练集中可能存在的错误标注（噪声标签）的容忍度。

### 3. 对抗权重校准 (Adversarial Weighting)
*   利用对抗验证（Adversarial Validation）计算训练集样本与测试集分布的相似度。
*   通过权重函数 $W = (\frac{p}{1-p})^{0.4}$ 对训练样本进行重新加权，使模型训练重心向测试集分布倾斜。

### 4. 自适应特征对齐 (Adaptive Alignment)
*   `test.py` 具备自动对齐功能，预测时会动态识别模型的特征需求，自动剔除或补全特征维度，确保推理阶段的特征空间与训练阶段 100% 一致。

