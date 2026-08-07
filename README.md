# DL-exercise

深度学习（Deep Learning）学习历程与实践代码，包含回归、分类、CNN 三个完整项目。

---

## 项目总览

| 项目 | 任务 | 模型 | 输入 |
|------|------|------|------|
| [Regression](#1-regression) | COVID-19 阳性率预测 | MLP | 93 维特征 |
| [Classification](#2-classification) | 帧级音素分类 | MLP | MFCC + 上下文拼接 |
| [CNN](#3-cnn) | 食物图像分类 | VGG 风格 CNN | 224x224 RGB |

---

## 1. Regression

**任务**：根据 COVID-19 统计数据预测测试阳性率（回归）。

**模型**：6 层全连接网络（93 → 32 → 64 → 128 → 64 → 32 → 1），ReLU 激活。

**训练配置**：

| 参数 | 值 |
|------|-----|
| 优化器 | Adam（lr=1e-3） |
| 损失函数 | MSELoss |
| Batch Size | 64 |
| Epochs | 3000（Early Stop=400） |
| 验证集比例 | 20% |

**数据**：`Regression/data/train.csv`（训练+验证）、`test.csv`（测试）。

**文件**：

```
Regression/
├── data/
│   ├── train.csv
│   └── test.csv
├── models/
│   └── best_model.pth
└── src/
    ├── train.py       # 训练脚本
    └── infer.py       # 推理脚本（输出 submission.csv）
```

---

## 2. Classification

**任务**：音素分类——给定音频的 MFCC 特征，对每一帧预测音素类别（41 类）。

**数据处理**：
- 原始特征：39 维 MFCC（13 系数 + 13 delta + 13 delta-delta）
- `concat_feat`：拼接前后帧的上下文信息（窗口 = 3，最终 117 维）
- `shift`：用边界值重复填充实现时序平移

**模型**：3 层全连接网络（117 → 256 → 256 → 41），ReLU 激活。

**训练配置**：

| 参数 | 值 |
|------|-----|
| 优化器 | AdamW（lr=1e-3） |
| 损失函数 | CrossEntropyLoss |
| Batch Size | 128 |
| Epochs | 10（Early Stop=3） |
| 训练/验证比例 | 80% / 20% |

**数据**：

```
Classification/data/libriphone/
├── train_split.txt       # 训练集文件列表（4286 条）
├── test_split.txt        # 测试集文件列表（1078 条）
├── train_labels.txt      # 帧级标签（每行：文件名 + N 个标签）
├── feat/
│   ├── train/            # 4286 个 .pt 文件（帧数, 39）
│   └── test/             # 1078 个 .pt 文件
└── sample_submission.csv # 提交模板
```

**文件**：

```
Classification/
├── data/...
├── models/
│   └── voice_model_best.pth
└── src/
    ├── train.py       # 训练脚本（含数据预处理、模型定义）
    └── infer.py       # 推理脚本（输出 submission.csv）
```

---

## 3. CNN

**任务**：食物图像分类——识别 11 种食物类别。

**模型**：VGG 风格的 5 层卷积 + 2 层全连接：

```
CNN:
  Conv(3→64) → BN → ReLU → MaxPool   → (64,  112, 112)
  Conv(64→128) → BN → ReLU → MaxPool  → (128,  56,  56)
  Conv(128→256) → BN → ReLU → MaxPool → (256,  28,  28)
  Conv(256→512) → BN → ReLU → MaxPool → (512,  14,  14)
  Conv(512→512) → BN → ReLU → MaxPool → (512,   7,   7)

Classifier:
  Flatten → Linear(25088→1024) → ReLU → Dropout(0.5) → Linear(1024→11)
```

**数据增强**（仅训练集）：

| 变换 | 参数 |
|------|------|
| Resize | (224, 224) |
| RandomHorizontalFlip | — |
| RandomRotation | ±10° |
| Normalize | mean=0.5, std=0.5 |

**训练配置**：

| 参数 | 值 |
|------|-----|
| 优化器 | AdamW（lr=1e-3） |
| 损失函数 | CrossEntropyLoss |
| Batch Size | 64 |
| Epochs | 30（Early Stop=3） |

**数据**：

```
CNN/data/food11/
├── training/       # 训练集（文件名格式：类别_编号.jpg）
├── validation/     # 验证集
└── test/           # 测试集
```

**文件**：

```
CNN/
├── data/...
├── models/
│   └── best_model.pth
└── src/
    ├── train.py       # 训练脚本（含数据增强、模型定义）
    └── infer.py       # 推理脚本（输出 submission.csv）
```

---

## 运行方式

所有脚本从各自 `src/` 目录下运行：

```bash
# Regression
cd Regression/src
python train.py
python infer.py

# Classification
cd Classification/src
python train.py
python infer.py

# CNN
cd CNN/src
python train.py
python infer.py
```

所有 `../data/` 和 `../models/` 路径均相对于 `src/` 目录。

---

## 环境依赖

- Python 3.8+
- PyTorch
- torchvision
- pandas, numpy
- tqdm, PIL

---

*随学习进度持续更新...*
