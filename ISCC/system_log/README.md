# 系统日志异常检测项目

本项目针对系统日志数据，构建了一个高性能的深度学习模型，旨在实现日志异常检测、异常类型识别（共 10 类）以及主异常区间的精确定位。目前该方案在本地验证及测试集上表现优异。

## 1. 项目目录结构
```text
项目根目录/
├─ 源码/                # 模型相关代码（含数据集、数据增强脚本等）
│  ├─ train.py          # 训练并保存模型脚本
│  ├─ test.py           # 加载模型并生成预测结果脚本
│  ├─ pseudo_label.py   # 伪标签生成脚本 (数据增强)
│  └─ data/             # 存放训练集与测试集原始 CSV
├─ 模型/                # 存放训练好的模型权重 (*.pth) 与词表 (vocab.pth)
├─ 提交结果/            # 存放生成的最终预测结果 (submission.csv)
├─ docker容器/           # Docker 镜像配置文件 (Dockerfile)
├─ requirements.txt      # 模型运行所需的全部环境依赖包
└─ README.md             # 运行说明与技术方案文档
```

## 2. 环境配置
建议使用 Python 3.10+ 环境。您可以直接使用 Docker 容器，或在本地安装依赖：
```bash
pip install -r requirements.txt
```

## 3. 运行说明

### 3.1 模型训练
运行 `源码/train.py` 开始训练。该脚本默认执行 5 折交叉验证，并自动保存每折的最佳模型至 `模型/` 目录。
```bash
python 源码/train.py
```

### 3.2 生成预测结果
运行 `源码/test.py` 加载 `模型/` 中的权重进行集成推理，生成的预测文件将保存于 `提交结果/submission.csv`。
```bash
python 源码/test.py
```

### 3.3 使用 Docker 运行 (推荐)
构建并运行镜像以确保环境一致性：
```bash
# 构建镜像 (在项目根目录下执行)
docker build -t log-detection -f docker容器/Dockerfile .

# 运行推理 (结果将输出到容器内的 /app/提交结果)
docker run --rm -v ${PWD}/提交结果:/app/提交结果 log-detection
```

### 3.4 使用虚拟环境运行 (备选方案)
若无法启动 Docker 环境，可在本地手动创建虚拟环境并安装依赖：

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

## 4. 模型架构与配置
本项目采用 **Multi-Scale CNN + Bi-LSTM + Transformer Encoder** 的混合架构：
- **词嵌入层 (Embedding)**：将日志词项映射为高维向量。
- **多尺度卷积 (Multi-Scale CNN)**：通过 [1, 3, 5, 7] 不同大小的卷积核提取行内局部语义特征。
- **双向 LSTM (Bi-LSTM)**：捕捉日志序列间的长距离依赖关系。
- **位置编码 (Positional Encoding)**：为 Transformer 引入显式行位置信息，增强对异常起止位置的敏感度。
- **Transformer Encoder**：通过自注意力机制进一步建模复杂的上下文关联。

## 5. 数据增强与核心优化策略
为了提升模型的泛化能力与定位精度，本项目采用了以下核心策略：

### 5.1 数据增强方法
1.  **伪标签学习 (Pseudo-Labeling)**：
    - 利用初步训练的模型对未标记的测试集进行预测。
    - 选取高置信度的预测结果（Top 10%）加入训练集进行二次训练，显著提升了模型在测试集分布上的表现。
2.  **对抗训练 (FGM)**：
    - 在 Embedding 层注入微小扰动进行训练，使模型对日志中的噪声（如变化的 ID 或时间戳）更具鲁棒性。

### 5.2 核心优化
1.  **标签平滑 (Label Smoothing)**：在损失函数中引入 0.1 的平滑因子，缓解模型对易混淆异常类的过拟合。
2.  **概率平均集成 (Ensemble)**：采用 5 折交叉验证模型，对输出概率进行加权平均，消除单模型过拟合风险。
3.  **精细化后处理**：
    - **空隙填充 (Gap Filling)**：自动合并间距小于 3 行的同类型异常区间，确保定位的连贯性。
    - **最小长度过滤 (MIN_LEN)**：通过网格搜索设定异常最小长度阈值为 2，过滤掉模型误报的孤立噪点。

## 6. 评分标准参考
- **最终得分** = $0.15 \cdot F1_{\text{detect}} + 0.50 \cdot \text{IoU}_{\text{loc}} + 0.35 \cdot F1_{\text{type}}$
- 本方案重点优化了 **IoU (区间定位)** 权重项，通过位置编码与后处理逻辑实现了高精度的起止点定位。
