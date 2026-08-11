# DL-exercise

深度学习（Deep Learning）学习历程与实践代码，包含回归、分类、CNN、Self-Attention、Transformer、GAN 六个完整项目。

---

## 项目总览

| 项目 | 任务 | 模型 | 输入 |
|------|------|------|------|
| [Regression](#1-regression) | COVID-19 阳性率预测 | MLP | 93 维特征 |
| [Classification](#2-classification) | 帧级音素分类 | MLP | MFCC + 上下文拼接 |
| [CNN](#3-cnn) | 食物图像分类 | VGG 风格 CNN | 224x224 RGB |
| [Self-Attention](#4-self-attention) | 说话人识别 | Transformer Encoder | mel 频谱 (T, 40) |
| [Transformer](#5-transformer) | 英中机器翻译 | Transformer (Encoder-Decoder) | 英文句子 → 中文句子 |
| [GAN](#6-gan) | 人脸图像生成 | DCGAN | 100 维随机噪声 → 64×64 RGB |

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

## 5. Transformer

**任务**：英中机器翻译（EN→ZH）——输入英文句子，输出中文翻译。

**数据**：TED2020 平行语料（394k 对），取前 150k 对训练。英文词级别分词（标点分离），词表 10000；中文逐字分词，词表 5000。特殊标记 `<pad>`, `<sos>`, `<eos>`, `<unk>` 占据 id 0~3。句子截断至 max_len=50。

**模型**：手写实现的标准 Transformer（无 `nn.Transformer` 依赖）。Encoder 和 Decoder 各 3 层，d_model=256、nhead=8、FFN 隐层 512。位置编码使用正弦/余弦方案。

核心组件自底向上手写：`scaled_dot_product_attention` → `MultiHeadAttention` → `FeedForward` → `EncoderLayer` / `DecoderLayer` → `TransformerNMT`。约 9M 参数。

- **Encoder**：词嵌入 + 位置编码 → 3×EncoderLayer（Self-Attention + FF，每子层带残差连接和 LayerNorm）。输出源句上下文表示 + padding mask。
- **Decoder**：词嵌入 + 位置编码 → 3×DecoderLayer（Masked Self-Attention + Cross-Attention + FF）。Cross-Attention 以 Decoder 状态为 Q、Encoder 输出为 K/V，完成源语言到目标语言的映射。
- **输出**：Linear 投影到中文词表（5000 维），softmax 取概率分布。
- **推理解码**：`greedy_decode` 自回归生成，从 `<sos>` 开始逐 token 预测，遇 `<eos>` 或达 max_len 停止。

**掩码策略**：Encoder 使用 padding mask `(B, 1, 1, L)` 屏蔽 `<pad>`；Decoder 自注意力使用因果掩码（上三角）+ padding mask 的联合掩码，确保位置 i 只能 attend 前面已生成的位置。

**学习率调度**：Warmup（前 10% step 从 0 线性升至 1e-4）+ Cosine 衰减至 0。配合 AdamW 优化器和梯度裁剪（max_norm=1.0）。

**损失函数**：CrossEntropyLoss，`ignore_index=PAD_IDX` 忽略 `<pad>` 位置，`label_smoothing=0.1` 标签平滑防过拟合。

**训练配置**：AdamW 优化器（lr=1e-4），CrossEntropyLoss，Batch Size 64，20 轮（Early Stop=5）。95% 训练 / 5% 验证划分，保存验证损失最低的模型权重。

**推理**：加载最佳模型对 `test.en` 批量翻译，贪心解码生成中文，输出到 `submission.zh`。

**文件结构**：
```
transformer/
├── data/
│   ├── ted2020/raw.en & raw.zh    # 训练数据（394k 对平行语料）
│   └── test/test.en               # 测试数据（4000 句英文）
├── models/
│   ├── vocab.pkl                  # 训练保存的英文/中文词表
│   └── transformer_nmt_best.pth   # 训练保存的最佳模型权重
└── src/
    ├── train.py                   # 训练脚本（含完整模型定义）
    └── infer.py                   # 推理脚本（批量翻译 + 输出 submission.zh）
```

---

## 6. GAN

**任务**：人脸图像生成——从 100 维随机噪声生成 64×64 的彩色人脸图像。

**数据**：从图片目录加载人脸，使用 `transform` 做 Resize(64,64)、RandomHorizontalFlip、Normalize 到 `[-1, 1]`，匹配 Generator 的 Tanh 输出范围。

**模型**：DCGAN（Deep Convolutional GAN），Generator 和 Discriminator 均为全卷积架构。

- **Generator**：噪声向量 `(100,)` → Linear 展开为 `(512, 4, 4)` 特征种子 → 4 次 `ConvTranspose2d`（kernel_size=5, stride=2, padding=2, output_padding=1）逐级放大至 `(3, 64, 64)`，每层接 BatchNorm + ReLU。最终层省略 BN，直接接 Tanh 将像素值约束在 `[-1, 1]`。通道数逐步衰减（512→256→128→64→3），空间尺寸逐步翻倍（4→8→16→32→64）。

- **Discriminator**：图像 `(3, 64, 64)` → 4 次 `Conv2d`（kernel_size=4, stride=2, padding=1）逐级降采样至 `(512, 4, 4)`，每层接 BatchNorm + LeakyReLU(0.2)。输入层省略 BN。通道数逐步递增（3→64→128→256→512）。最后一层 Conv2d(k=4, s=1, p=0) 将 4×4 特征图压为 `(1, 1, 1)` 标量，Sigmoid 输出 `[0, 1]` 表示真实概率。

- **权重初始化**：均值 0、标准差 0.02 的正态分布（DCGAN 论文推荐）。

**训练策略**：
- **损失函数**：BCELoss。标签平滑——真实图标签用 0.9 代替 1.0，防止 D 过度自信导致梯度过早消失。假图标签保持 0。
- **对抗博弈**：每 batch 先训 D（分辨真/假图），再训 G（试图骗过 D）。G 每步训 2 次（不同噪声），给 Generator 更多追赶机会。
- **批量归一化**：Generator 全层接 BN；Discriminator 除输入层外均接 BN。最后一层均省略 BN。WGAN-GP 兼容性：代码注释提醒改用 InstanceNorm2d。

**优化器**：Adam，lr=0.0002，beta1=0.5（低于默认 0.9，GAN 训练需更灵敏的梯度响应），beta2=0.999。

**训练配置**：Batch Size 64，15 轮。每 5 轮保存一次模型权重 + 固定噪声生成的样本图，训练结束保存 G_final.pth。约 9M 参数（G: ~5.7M，D: ~3.3M）。

**推理**：加载 G_final.pth，输入随机噪声批量生成 64 张人脸，拼成 8×8 网格保存。

**文件结构**：
```
GAN/
├── data/faces/                    # 人脸图像目录
├── models/
│   ├── G_epoch_5.pth              # 每 5 轮保存的 Generator 权重
│   ├── D_epoch_5.pth              # 每 5 轮保存的 Discriminator 权重
│   └── G_final.pth                # 最终 Generator 权重
├── output/
│   ├── epoch_005.png              # 每 5 轮生成的样本图
│   └── inference_result.png       # 推理生成的 8×8 网格
└── src/
    ├── train.py                   # 训练脚本（含 Generator/Discriminator 定义）
    └── infer.py                   # 推理脚本（加载 G 生成图像）
```

---

## 环境依赖

- Python 3.8+
- PyTorch + torchvision
- pandas, numpy
- tqdm, PIL
