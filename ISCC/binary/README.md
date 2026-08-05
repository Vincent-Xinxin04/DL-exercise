# 二进制漏洞检测与分类项目说明文档

本项目针对二进制可执行文件进行漏洞检测（Task 1）与漏洞类型分类（Task 2）。系统采用深度学习模型 1D-ResNet 结合 PE 元数据特征，通过多阶段训练策略实现高精度的漏洞识别。

## 1. 项目目录结构

按照赛题要求，项目组织结构如下：

```text
项目根目录/
├─ 源码/                # 模型实现、训练与推理脚本
│  ├─ data/             # 数据存放目录 (需自行放入 train.csv 等)
│  ├─ dataset.py        # 数据集加载逻辑
│  ├─ model.py          # 1D-ResNet + MLP 模型架构
│  ├─ preprocess.py     # 特征提取工具 (指令序列 + PE 元数据)
│  ├─ train.py          # 训练全流程调度脚本
│  ├─ train_stage1.py   # 基础模型训练脚本
│  ├─ train_stage2.py   # HEM 强化与增强微调脚本
│  └─ test.py           # 推理与生成提交结果脚本
├─ 模型/                # 存放训练好的模型权重 (*.pth) 及特征缓存
├─ 提交结果/            # 存放生成的最终预测文件 (submission.csv)
├─ docker容器/           # Docker 部署配置文件
│  └─ Dockerfile        # 容器镜像构建脚本
├─ requirements.txt      # Python 环境依赖列表
└─ README.md             # 本说明文档
```

## 2. 环境配置

### 本地环境
建议使用 Python 3.9+。
1. **安装依赖**：
   ```bash
   pip install -r requirements.txt
   ```
2. **硬件要求**：建议使用支持 CUDA 的 NVIDIA GPU 以加速训练。

### Docker 环境
项目提供了预配置的 Docker 环境，确保运行环境的一致性。
1. **构建镜像**（在根目录下执行）：
   ```bash
   docker build -t vuln-detection -f docker容器/Dockerfile .
   ```
2. **运行容器**：
   ```bash
   docker run --gpus all vuln-detection
   ```

### 虚拟环境（备选方案）
若无法启动 Docker 环境，可手动创建 Python 虚拟环境运行项目：

1. **创建虚拟环境**（在项目根目录执行）：
   ```bash
   python -m venv venv
   ```

2. **激活虚拟环境**：
   - Windows：
     ```powershell
     .\venv\Scripts\activate
     ```
   - Linux/Mac：
     ```bash
     source venv/bin/activate
     ```

3. **安装依赖**：
   ```bash
   pip install -r requirements.txt
   ```

4. **验证安装**：
   ```bash
   python -c "import torch; print(torch.__version__)"
   ```

5. **运行预测**：
   ```bash
   python 源码/test.py
   ```

**注意事项**：
- 确保已安装 Python 3.9+ 版本
- 建议使用支持 CUDA 的 GPU 以加速推理
- 虚拟环境激活后，命令行提示符会显示 `(venv)` 前缀，表示已进入虚拟环境

## 3. 运行方式 (复现步骤)

### 第一步：准备数据
将比赛数据集放入 `源码/data/` 目录下，确保包含 `train.csv`、`test.csv` 及对应的二进制文件。

### 第二步：模型训练
运行一键训练脚本，该脚本会自动执行“基础训练 -> 伪标签生成 -> HEM 强化微调”全流程：
```bash
python 源码/train.py
```
训练完成后，最终模型将保存于 `模型/final_model.pth`。

### 第三步：推理生成结果
执行推理脚本加载最终模型并生成符合官方格式的 `submission.csv`：
```bash
python 源码/test.py
```
结果文件将存放在 `提交结果/` 目录中。

## 4. 算法与数据增强说明

### 模型架构
- **指令序列分支 (1D-ResNet)**：对二进制文件的原始指令字节进行嵌入（Embedding），通过多层一维残差网络提取序列特征，感知指令间的依赖关系。
- **元数据分支 (MLP)**：利用 `pefile` 提取 PE 文件的节区数量、入口点位置、导入/导出函数量及信息熵等 10 维关键元数据，辅助模型判断文件性质。
- **多任务学习**：模型末端分为检测头（2类）和分类头（86类），通过联合损失函数（Joint Loss）同步优化。

### 核心策略
1. **半监督伪标签增强 (Pseudo-labeling)**：
   在 Stage 1 结束后，利用基础模型对测试集进行预测，将高置信度的预测结果作为“伪标签”数据反馈给训练集。此方法能有效对齐训练集与测试集的分布差异，提升泛化能力。
2. **自动难样本挖掘 (HEM - Hard Example Mining)**：
   在 Stage 2 强化阶段，系统计算样本 Loss 值，针对 Loss 较高的“难分类样本”进行重采样和针对性学习，显著提升了模型对长尾分布（少数类 CWE）的识别准确率。
3. **加权 Focal Loss**：
   针对 86 类 CWE 标签存在严重类别不平衡的情况，采用了带类别权重的 Focal Loss，迫使模型关注样本稀少的漏洞类型。

## 5. 提交要求合规性
- **文件编码**：输出的 `submission.csv` 采用 UTF-8 编码。
- **表头格式**：包含 `binary_id`, `label`, `cwe_id` 三列。
- **标签映射**：严格遵循 `question.md` 定义的 0/1 标签及 CWE 编号规范。
