# DL-exercise

深度学习（Deep Learning）学习历程与实践代码，包含回归、分类、CNN、Self-Attention 四个完整项目。

---

## 项目总览

| 项目 | 任务 | 模型 | 输入 |
|------|------|------|------|
| [Regression](#1-regression) | COVID-19 阳性率预测 | MLP | 93 维特征 |
| [Classification](#2-classification) | 帧级音素分类 | MLP | MFCC + 上下文拼接 |
| [CNN](#3-cnn) | 食物图像分类 | VGG 风格 CNN | 224x224 RGB |
| [Self-Attention](#4-self-attention) | 说话人识别 | Transformer Encoder | mel 频谱 (T, 40) |

---

## 1. Regression

**任务**：根据 COVID-19 统计数据预测测试阳性率（回归）。

**模型**：6 层全连接网络（93 → 32 → 64 → 128 → 64 → 32 → 1），ReLU 激活。

**训练配置**：Adam 优化器（lr=1e-3），MSELoss，Batch Size 64，训练 3000 轮（Early Stop=400）。

**数据**：CSV 表格，93 维统计特征 + 目标值。训练集 / 验证集按 80% / 20% 拆分。

---

## 2. Classification

**任务**：音素分类——给定音频的 MFCC 特征，对每一帧预测音素类别（41 类）。

**数据处理**：39 维 MFCC（13 系数 + 13 delta + 13 delta-delta），通过 `concat_feat` 拼接前后帧上下文（窗口 = 3），最终 117 维。`shift` 函数用边界值重复填充实现时序平移。

**模型**：3 层全连接网络（117 → 256 → 256 → 41），ReLU 激活。

**训练配置**：AdamW 优化器（lr=1e-3），CrossEntropyLoss，Batch Size 128，10 轮（Early Stop=3）。

**数据**：LibriPhone 数据集，4286 条训练音频 + 1078 条测试音频，每帧一个音素标签（0~40）。

---

## 3. CNN

**任务**：食物图像分类——识别 11 种食物类别。

**模型**：VGG 风格 5 层卷积 + 2 层全连接。通道数逐步递增（3 → 64 → 128 → 256 → 512 → 512），每层接 BatchNorm + ReLU + MaxPool（尺寸减半）。分类头为 Flatten → 25088 → 1024 → 11，含 Dropout(0.5) 防过拟合。

**数据增强**：Resize(224,224)、RandomHorizontalFlip、RandomRotation(±10°)、Normalize。仅训练集使用，验证/测试集不做增强。

**训练配置**：AdamW 优化器（lr=1e-3），CrossEntropyLoss，Batch Size 64，30 轮（Early Stop=3）。

---

## 4. Self-Attention

**任务**：说话人识别——给定语音的 mel 频谱特征，预测说话人身份（600 类）。

**模型**：Transformer Encoder（无 Decoder）。mel 频谱（40 维）经 Linear 升维至 d_model=80，通过 6 层 TransformerEncoder（nhead=8, FFN=2048）提取帧间依赖，mean pooling 汇总为声纹向量，最后全连接分类到 600 人。

**学习率调度**：Warmup + Cosine——前 10% 步数 lr 从 0 线性升至 1e-3，之后余弦衰减至 0。Transformer 无 warmup 难以收敛，这是训练关键。

**数据**：训练时随机截取 256 帧（数据增强），推理时截取前 256 帧。每条音频 40 维 mel 频谱，通过 `metadata.json` 索引说话人标签。

**训练配置**：Adam 优化器（lr=1e-3），CrossEntropyLoss，Batch Size 32，15 轮（Early Stop=3）。

---

## 环境依赖

- Python 3.8+
- PyTorch + torchvision
- pandas, numpy
- tqdm, PIL
