# DL-exercise

深度学习（Deep Learning）学习历程与实践代码，包含Regression、Classification、CNN、Self-Attention、Transformer、GAN、BERT、Autoencoder、RL 9个基础项目，以及 ISCC 竞赛平台的四个安全AI子项目。

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
| [BERT](#7-bert) | 中文问答（抽述式） | BERT-base-chinese | 段落 + 问题 → 答案片段 |
| [Autoencoder](#8-autoencoder) | 人脸异常检测 | VAE / ConvAutoencoder | 64×64 RGB → 重建误差 |
| [RL](#9-rl) | 月球着陆器控制 | REINFORCE (Policy Gradient) | 8 维状态 → 4 维动作 |
| [ISCC](#10-iscc) | 安全AI竞赛（4题） | 1D-ResNet / LightGBM / XGBoost+LGB+CatBoost / MultiScale-CNN+BiLSTM+Transformer | 二进制 / 流量 / PowerShell / 系统日志 |

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

## 7. BERT

**任务**：中文抽述式问答（Extractive QA）——给定段落和问题，从段落中抽取答案片段。

**数据**：31,690 训练 / 4,131 验证 / 4,957 测试条 QA 对。每条包含段落文本（`paragraphs`）+ 问题列表（`questions`），标注了答案的起止字符位置（`answer_start`/`answer_end`）和答案文本（`answer_text`）。同一段落可能关联多条问题。

**模型**：`bert-base-chinese`（12 层 Transformer Encoder，768 维，110M 参数）微调。`BertForQuestionAnswering` 在 BERT 输出之上添加两个独立 Linear 头，分别预测答案起始位置和结束位置的 token 级概率。调整 logits 逐窗口比较 `start_prob + end_prob` 选取最佳答案。

长段落使用滑窗（sliding window）处理，每个窗口独立预测，最后选置信度最高的窗口的答案。

**数据处理**：
- 问题和段落分别 tokenize（`add_special_tokens=False`），在 `__getitem__` 中拼接 `[CLS]` + question + `[SEP]` + paragraph_window + `[SEP]`，max_seq_len = 193。
- **训练**：以答案中心为锚点，截取固定窗口（window_len=150），确保答案完整落入窗口内。答案坐标需从字符位置转换为 token 位置（`char_to_token`），再映射到拼接后序列中的新位置。
- **验证 / 测试**：`doc_stride=150` 滑动窗口覆盖整段，每条 QA 对可能产生多个窗口。

**训练策略**：
- 混合精度训练（AMP）：`torch.amp.autocast('cuda')` + `GradScaler`，batch_size=32。
- 优化器：`torch.optim.AdamW`，lr=5e-5，5 轮训练（Early Stop patience=3）。
- 验证指标：Exact Match（EM），模型预测答案与真实答案文本完全匹配才算正确。

**推理**：加载最佳模型权重，对测试集逐条滑动窗口预测，输出 `{id, answer}` 列表保存为 JSON。

**文件结构**：
```
Bert/
├── data/
│   ├── hw7_train.json             # 训练集（31,690 QA 对）
│   ├── hw7_dev.json               # 验证集（4,131 QA 对）
│   ├── hw7_test.json              # 测试集（4,957 QA 对）
│   └── hw7_test_result.json       # 推理输出
├── models/
│   └── bert_qa_best.pth           # 最佳模型权重
└── src/
    ├── train.py                   # 训练脚本（含 Dataset、evaluate、训练循环）
    └── infer.py                   # 推理脚本（加载模型、滑动窗口预测）
```

---

## 8. Autoencoder

**任务**：无监督异常检测——训练集全部为正常人脸图像，测试集混有异常图像，通过重建误差识别异常。

**数据**：`trainingset.npy`（100,000 张正常人脸，64×64×3，uint8）+ `testingset.npy`（19,636 张，约半数异常）。数据以 numpy 数组形式存储，NHWC 格式，像素值 [0, 255]。

**数据处理**：加载后 permute 为 NCHW 格式，归一化到 `[-1, 1]` 匹配 Tanh 输出范围（`2*x/255 - 1`）。

**模型**：实现三种自编码器，通过 `config["model_idx"]` 切换。

- **全连接 Autoencoder**：Flatten(12288) → 6 层全连接压缩至 4 维 latent → 6 层全连接还原至 12248 维，ReLU 激活，输出层 Tanh。
- **ConvAutoencoder**：3 层 `Conv2d`（stride=2, 通道 3→16→32→64）下采样至 8×8×64 latent → 3 层 `ConvTranspose2d`（output_padding=1 精确还原尺寸）上采样回 64×64×3。
- **VAE（Variational Autoencoder）**：编码器分叉输出 `mu` 和 `logvar`，通过重参数化技巧 `z = mu + eps * std` 采样 latent，再解码重建。latent 为概率分布而非确定点，具备生成能力。

**损失函数**：VAE 使用重建损失（MSE）+ KL 散度（`-0.5 * Σ(1 + logvar - mu² - exp(logvar))`），强迫 latent 分布接近标准正态 `N(0, 1)`。

**训练配置**：Adam 优化器（lr=1e-3, weight_decay=1e-5），Batch Size 32，30 轮，保存训练损失最低的模型权重。

**推理**：对测试集每张图计算重建误差（MSE），作为异常得分输出到 CSV（`ID, score` 格式，19,636 行）。得分越高越可能为异常，由评分服务器用隐藏标签计算 ROC AUC。

**文件结构**：
```
Autoencoder/
├── data/
│   ├── trainingset.npy             # 训练集（100,000 张正常人脸）
│   └── testingset.npy              # 测试集（19,636 张，含异常）
├── models/
│   └── autoencoder_best.pth        # 最佳模型权重
├── output/
│   └── anomaly_scores.csv          # 异常得分输出（提交格式）
└── src/
    ├── train.py                    # 训练脚本（含三种模型定义）
    └── infer.py                    # 推理脚本（重建误差 → 异常得分）
```

---

## 9. RL

**任务**：月球着陆器控制——训练智能体在 LunarLander-v3 环境中实现安全着陆。

**算法**：REINFORCE（策略梯度蒙特卡洛方法），属于强化学习中最基础的 on-policy 策略梯度算法。

**模型**：三层全连接策略网络（8 → 16 → 16 → 4），tanh 隐藏层激活 + softmax 输出动作概率分布。

**核心组件**：
- **策略网络（Policy Network）**：将 8 维状态映射为 4 个动作的概率分布
- **Agent**：封装策略网络，提供动作采样（探索）和策略更新能力
- **折扣因子（γ=0.99）**：计算折扣累计回报，平衡即时与长远奖励
- **奖励标准化**：对多条轨迹的回报做 Z-score 归一化，降低梯度估计方差

**训练流程**：
1. 每个 epoch 收集 5 条轨迹（共 5 个 episode），记录每步的 log_prob 和 reward
2. 反向计算每步的折扣累计回报 G_t = r_t + γ·r_{t+1} + ...
3. 所有轨迹的回报一起标准化，得到 advantage
4. 使用 REINFORCE 损失 loss = -Σ log π(a_t|s_t) × G_t 更新策略网络

**训练配置**：SGD 优化器（lr=0.001），500 个 epoch（每 epoch 收集 5 条轨迹），种子 543。

**环境**：LunarLander-v3（gymnasium）。

- **状态空间**：8 维连续向量（位置 x/y、速度 x/y、角度、角速度、左右腿接触标志）
- **动作空间**：4 个离散动作（不动 / 左推进 / 主引擎 / 右推进）
- **奖励规则**：接近降落点 + 分，主引擎点火 -0.3/帧，接触地面 +10，成功着陆 +200，坠毁 -100

**推理**：加载训练好的策略网络，使用贪婪策略（argmax）进行 5 次测试，生成动作序列用于提交。

**文件结构**：
```
RL/
├── src/
│   ├── train.py                   # 训练脚本（含网络和 Agent 定义）
│   └── infer.py                   # 推理脚本（加载模型、生成动作序列）
├── models/
│   └── policy_gradient.pth        # 训练好的策略网络权重
├── output/
│   ├── total_rewards.png          # 累计奖励训练曲线
│   ├── final_rewards.png          # 最终奖励训练曲线
│   └── Action_List.npy            # 测试动作序列（提交格式）
└── README.md
```

---

## 10. ISCC

ISCC（信息安全常识竞赛）安全AI竞赛项目，包含四个子赛题，覆盖二进制漏洞检测、网络流量分类、PowerShell 恶意脚本识别和系统日志异常检测。

### 10.1 Binary — 二进制漏洞检测与分类

**任务**：对二进制可执行文件进行漏洞检测（2 类）与漏洞类型分类（86 类 CWE）。

**模型**：1D-ResNet（指令序列分支）+ MLP（PE 元数据分支）多任务架构。指令字节经 Embedding → 1D 残差网络提取序列特征；`pefile` 提取 10 维 PE 元数据辅助判断。模型末端分叉为检测头（2 类）和分类头（86 类），联合损失优化。

**核心策略**：
- **半监督伪标签增强**：基础模型对测试集生成高置信度伪标签，反馈训练集缩小分布差异。
- **HEM 难样本挖掘**：针对长尾类别（稀有 CWE）重采样高 Loss 样本，提升少样本类识别率。
- **加权 Focal Loss**：类别不平衡时强制模型关注样本稀少的漏洞类型。

**训练配置**：PyTorch，Adam 优化器，Focal Loss，分 Stage1（基础训练）→ Stage2（HEM 强化微调）两阶段训练。

**文件结构**：
```
ISCC/binary/
├── 源码/
│   ├── data/                   # 训练/测试数据
│   ├── dataset.py              # 数据集加载
│   ├── model.py                # 1D-ResNet + MLP 模型
│   ├── preprocess.py           # 指令序列 + PE 元数据特征提取
│   ├── train.py                # 一键训练（基础→伪标签→HEM）
│   ├── train_stage1.py         # 基础模型训练
│   ├── train_stage2.py         # HEM 强化微调
│   └── test.py                 # 推理 → submission.csv
├── 模型/                       # 模型权重 (.pth) + 特征缓存 (.pkl)
├── 提交结果/submission.csv
└── docker容器/Dockerfile
```

---

### 10.2 Net Classification — 网络流量安全事件类别判别

**任务**：在强漂移环境下对网络流量进行安全事件类别判别。

**模型**：LightGBM 集成（3 种子 × 5 折交叉验证 = 15 个子模型）。

**核心策略（V9 提纯版）**：
- **特征提纯**：移除冗余分桶特征，保留 Top-10 重要特征的平方项与交互项，捕捉非线性关系。
- **高斯噪声注入**：训练时注入 Scale=0.05 的高斯噪声，防止过拟合。
- **对抗权重校准**：通过对抗验证计算训练/测试分布相似度，按权重函数 $W = (p/(1-p))^{0.4}$ 重加权训练样本。
- **自适应特征对齐**：推理时自动识别模型所需特征，动态补全/剔除维度。

**训练配置**：LightGBM，5 折 CV，3 种随机种子（42 / 1024 / 2026），标签平滑 0.1。

**文件结构**：
```
ISCC/net_classification/
├── 源码/
│   ├── data/                   # train_data.csv / test_data.csv
│   ├── train.py                # 特征工程 + 对抗校准 + 集成训练
│   ├── test.py                 # 加载集成模型 → 预测
│   └── compare.py              # 模型对比工具
├── 模型/                       # 15 个 .joblib 模型 + top_features.joblib
├── 提交结果/submission.csv
└── docker容器/Dockerfile
```

---

### 10.3 PowerShell — PowerShell 恶意脚本分类

**任务**：将 PowerShell 脚本分为 3 类：正常脚本 / 一般恶意脚本 / 混淆恶意脚本。

**模型**：XGBoost + LightGBM + CatBoost 三模型加权集成，3 种子 × 5 折 = 45 个子模型概率平均。

**核心策略**：
- **高级特征工程**：行为强度统计、解码×网络交互、混淆深度（布局变化率、编码率）、作用域复杂度分析。
- **防泄露 Target Encoding**：K-Fold 目标编码 + 轻微噪声，防止线下/线上分数偏差。
- **类别加权**：对恶意类（1、2）设置 1.2 倍损失权重，提升 Macro-F1。

**训练配置**：XGBoost / LightGBM / CatBoost，5 折 CV，3 种子平均，OOF 搜索最优融合权重。

**文件结构**：
```
ISCC/powershell/
├── 源码/
│   ├── data/                   # data_train.csv / data_test.csv
│   ├── train.py                # 特征工程 + 交叉验证 + 集成训练
│   └── test.py                 # 加载集成模型 → 预测
├── 模型/                       # .json / .txt / .cbm / .npy 模型文件
├── 提交结果/submission.csv
└── docker容器/Dockerfile
```

---

### 10.4 System Log — 系统日志异常检测

**任务**：系统日志异常检测，需识别 10 类异常类型并精确定位异常起止区间。

**模型**：Multi-Scale CNN + Bi-LSTM + Transformer Encoder 混合架构。

- **词嵌入层**：日志词项映射为高维向量。
- **多尺度 CNN**：卷积核 [1, 3, 5, 7] 提取行内局部语义特征。
- **Bi-LSTM**：捕捉日志序列间长距离依赖。
- **位置编码**：显式位置信息增强异常起止敏感度。
- **Transformer Encoder**：自注意力建模复杂上下文关联。

**核心策略**：
- **伪标签学习**：初步模型对测试集 Top 10% 高置信度预测作为伪标签反馈训练。
- **对抗训练（FGM）**：Embedding 层注入微小扰动，增强对日志噪声的鲁棒性。
- **精细化后处理**：空隙填充（合并间距 < 3 行的同类型区间）+ 最小长度过滤（阈值 2 行）。

**评分公式**：$0.15 \cdot F1_{detect} + 0.50 \cdot IoU_{loc} + 0.35 \cdot F1_{type}$，重点优化区间定位（IoU）。

**训练配置**：PyTorch，5 折 CV，每折保存最佳权重，推理时 5 模型概率平均集成。

**文件结构**：
```
ISCC/system_log/
├── 源码/
│   ├── data/                   # train.csv / test.csv / test_data_b.csv
│   ├── train.py                # 5 折交叉验证训练
│   ├── test.py                 # 集成推理 + 后处理
│   ├── pseudo_label.py         # 伪标签生成
│   ├── eda.py                  # 探索性数据分析
│   └── compare.py              # 模型对比
├── 模型/                       # 5 折 .pth 权重 + vocab.pth
├── 提交结果/submission.csv
└── docker容器/Dockerfile
```

---

## 环境依赖

- Python 3.8+
- PyTorch + torchvision
- pandas, numpy
- tqdm, PIL
- transformers (HuggingFace)
- gymnasium[box2d], pygame (RL 项目)
