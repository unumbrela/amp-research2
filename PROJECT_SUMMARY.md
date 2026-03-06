# ESM-DiffVAE 项目总结

## 一、项目概述

**AMP Forge** 是一个通用抗菌肽 (Antimicrobial Peptide, AMP) 生成项目，核心目标是构建"可用、可扩展、可验证"的 AMP 序列设计工作流。项目基于 **ESM-DiffVAE v8** 架构——一个融合蛋白质语言模型 (PLM) 表征、变分自编码器 (VAE) 和潜在空间扩散模型 (Latent Diffusion) 的深度生成模型。

**仓库地址**: GitHub Pages 展示页 — https://unumbrela.github.io/amp-research2/

---

## 二、核心架构: ESM-DiffVAE v8

整体模型由以下组件构成，形成一个完整的"编码 → 潜在空间建模 → 解码"流水线：

```
输入序列 ──→ [PLM 特征提取] ──→ [AA 编码] ──→ [BiGRU Encoder] ──→ mu, sigma ──→ z
                                                                              |
                                              [Latent Diffusion] <──── z (训练/采样)
                                                                              |
                                   [Non-AR Transformer Decoder] <── z + properties
                                                                              |
                                                              生成的 AMP 序列 (并行输出)
```

### 2.1 蛋白质语言模型 (PLM) 特征提取器

**文件**: `esm_diffvae/models/plm_extractor.py`

统一接口支持三种 PLM 后端，均冻结参数用于提取 per-residue 嵌入：

| 后端 | 模型 | 嵌入维度 |
|------|------|---------|
| ESM-2 | `esm2_t6_8M_UR50D` / `esm2_t12_35M_UR50D` | 320 / 480 |
| Ankh | `ankh-base` / `ankh-large` | 768 / 1536 |
| ProtT5 | `prot_t5_xl_half` | 1024 |

当前配置默认使用 **ProtT5** (`prot_t5_xl_half`, 1024维)。

### 2.2 氨基酸编码

**文件**: `esm_diffvae/models/aa_encoding.py`

采用 **BLOSUM62 + 可学习嵌入** 的混合编码方案，替代传统 one-hot 编码：

- **BLOSUM62 (20维)**: 固定的进化替换矩阵得分，编码氨基酸间的生化相似性
- **Learnable Embedding (16维)**: 端到端学习的任务特定特征
- **合计输出**: 36 维 / 残基

### 2.3 编码器 (Encoder)

**文件**: `esm_diffvae/models/encoder.py`

- **结构**: 双向 GRU (Bidirectional GRU)
- **输入**: PLM 嵌入 + AA 编码拼接 → 渐进式线性投影 (避免过大的维度瓶颈) → BiGRU
- **输出**: 前向/后向隐状态拼接 → 映射到潜在分布参数 (mu, logvar)
- **配置**: hidden_dim=256, latent_dim=64, 1层 BiGRU, dropout=0.3

### 2.4 解码器 (Decoder)

**文件**: `esm_diffvae/models/decoder.py`

- **结构**: **非自回归 Transformer 解码器**，并行预测所有氨基酸位置
- **关键设计**:
  - 条件向量 (z + properties) 经 2 层 MLP 投影后广播到所有位置，加上可学习位置嵌入
  - **Pre-LayerNorm Transformer blocks**: 自注意力为每个位置提供全局上下文
  - 非自回归设计消除了暴露偏差 (exposure bias) 和误差累积问题
  - 模型较小 (~350K params)，适配 6.6K 样本的数据集规模
- **配置**: decoder_hidden_dim=128, 3层 Transformer, n_heads=4, ffn_dim=256, dropout=0.3
- **长度预测**: 从 z 直接预测序列长度分布

### 2.5 潜在空间扩散 (Latent Diffusion)

**文件**: `esm_diffvae/models/latent_diffusion.py`

在 VAE 的 64 维潜在空间中运行高斯扩散过程：

- **去噪网络**: MLP (DenoiserMLP)，含时间嵌入和属性嵌入
- **扩散步数**: T=50, cosine 噪声调度
- **Classifier-Free Guidance (CFG)**: 训练时以 10% 概率丢弃属性条件，推理时支持引导尺度
- **功能**:
  - `sample()`: 从噪声完全去噪 → **无条件生成**
  - `partial_denoise()`: 对输入 z 部分加噪再去噪 → **变体生成**

### 2.6 属性预测头

**文件**: `esm_diffvae/models/property_heads.py`

从潜在向量 z 预测多种肽属性，用于辅助训练和条件生成：

| 属性 | 类型 | 说明 |
|------|------|------|
| `is_amp` | 二分类 | 是否为抗菌肽 |
| `mic_value` | 回归 | 最小抑菌浓度 |
| `is_toxic` | 二分类 | 是否有毒性 |
| `is_hemolytic` | 二分类 | 是否溶血 |
| `length_norm` | — | 归一化长度 |

### 2.7 判别器 (Discriminator)

**文件**: `esm_diffvae/models/discriminator.py`

- **结构**: BiGRU 判别器 (Embedding → BiGRU → FC → sigmoid)
- **用途**: RL 微调阶段 (Phase 1B) 区分真实序列与生成序列
- **设计**: 刻意保持小规模 (~30K params)，避免 mode collapse

---

## 三、训练流程

分为多个阶段依次执行：

### Phase 1A: 数据准备与 VAE 预训练

#### 数据准备
```bash
python data/prepare_data.py --min-len 5 --max-len 50
```
过滤并划分数据集 (train 80% / val 10% / test 10%)。

#### PLM 嵌入预计算
```bash
python data/compute_embeddings.py --backend prot_t5 --model prot_t5_xl_half
```
使用选定 PLM 后端对所有序列提取嵌入，保存为 `.pt` 文件，避免训练时重复计算。

#### VAE MLE 训练
**文件**: `esm_diffvae/training/train_vae.py`

- **损失函数** (`esm_diffvae/training/losses.py`):
  - 重建损失 (Cross-Entropy, label smoothing=0.02)
  - KL 散度 (Cyclical Annealing, 2 cycles, free bits=0.1)
  - 监督对比损失 (SupCon, 当前 lambda=0.0, 已禁用)
  - 属性预测损失 (当前 lambda=0.0, 已禁用——因 mic/toxic/hemolytic 数据缺失)
  - 长度预测损失 (lambda=0.2)
- **优化**: AdamW + Cosine LR + FP16 混合精度 + EMA (decay=0.999)
- **配置**: 300 epochs, batch_size=64, lr=3e-4, beta_max=1.0, 梯度裁剪=1.0
- **早停**: patience=40 epochs

### Phase 1B: RL 微调 (对抗式精炼)

**文件**: `esm_diffvae/training/train_vae_rl.py`

- 加载 Phase 1A 预训练的 VAE checkpoint，冻结编码器
- 使用 BiGRU 判别器 + REINFORCE 策略梯度微调解码器
- **配置**: 50 epochs, lr=1e-5, disc_lr=1e-4, rl_weight=0.5, disc_steps_per_gen_step=3

### Phase 2: 扩散模型训练
**文件**: `esm_diffvae/training/train_diffusion.py`

- 冻结 VAE，先用编码器将全部训练数据编码为潜在向量
- 在潜在空间上训练去噪网络 (MSE loss)
- **配置**: 500 epochs, batch_size=128, lr=1e-4
- 最终保存完整模型 `esm_diffvae_full.pt`

---

## 四、序列生成

### 4.1 无条件生成 (De Novo)

**文件**: `esm_diffvae/generation/unconditional.py`

从扩散先验直接采样全新 AMP 序列：

```bash
python generation/unconditional.py \
  --checkpoint checkpoints/esm_diffvae_full.pt \
  --n-samples 100 --top-p 0.9
```

流程: 纯噪声 → Diffusion 去噪 → z → 长度预测 → 非自回归并行解码 → AMP 序列

支持 Nucleus Sampling (top-p) 和温度调节。

### 4.2 条件变体生成

**文件**: `esm_diffvae/generation/variant.py`

以已有肽序列为母体，生成结构可控的变体。支持 **5 种变异模式**：

| 模式 | 说明 | 配置比例 |
|------|------|---------|
| `c_sub` | C端替换：保留前 K 个 AA，重新生成最后 N 个位置 | 40% |
| `c_ext` | C端延伸：保留母序列，在末端追加 1-5 个新 AA | 20% |
| `c_trunc` | C端截断重生：截掉末尾若干 AA，从截断点重新生长 | 15% |
| `tag` | 标签追加：在 C端添加常见肽标签 (His6, FLAG 等) | 10% |
| `latent` | 潜在扰动：在潜在空间中微扰 z 后解码 (全序列级变化) | 15% |

```bash
python generation/variant.py \
  --checkpoint checkpoints/esm_diffvae_full.pt \
  --input-sequence "GIGKFLHSAKKFGKAFVGEIMNS" \
  --mode mixed --n-variants 50
```

核心技术: **非自回归解码 + 前缀拼接** — 解码器并行输出完整序列，然后将母序列的 N 端前缀与生成序列的 C 端后缀拼接，实现对 N 端保守 / C 端变异的控制。

### 4.3 序列插值

**文件**: `esm_diffvae/generation/interpolation.py`

在两条序列的潜在空间之间进行球面线性插值 (SLERP)，生成平滑过渡的序列家族。

---

## 五、评估体系

### 5.1 评估指标

**文件**: `esm_diffvae/evaluation/metrics.py`

| 类别 | 指标 |
|------|------|
| 基础统计 | 序列长度分布、唯一性比率 |
| 组成分析 | AA 频率分布、与天然 AMP 的 KL 散度 |
| 多样性 | 成对编辑距离（归一化） |
| 新颖性 | 与训练集的重叠率 |
| 物化性质 | 净电荷分布、疏水残基比例 |
| 变体指标 | 序列同一性、编辑距离分布 |

### 5.2 理化验证

**文件**: `esm_diffvae/evaluation/physicochemical.py`

### 5.3 计算验证

**文件**: `esm_diffvae/evaluation/computational_validation.py`

### 5.4 可视化

**文件**: `esm_diffvae/evaluation/visualization.py`

---

## 六、项目结构

```
esm_diffvae/
├── configs/
│   └── default.yaml              # 全局配置文件
├── models/
│   ├── esm_diffvae.py            # 主模型类 (整合所有组件)
│   ├── plm_extractor.py          # 多 PLM 后端特征提取器
│   ├── aa_encoding.py            # BLOSUM62 + 可学习嵌入
│   ├── encoder.py                # BiGRU VAE 编码器
│   ├── decoder.py                # 非自回归 Transformer 解码器
│   ├── discriminator.py          # BiGRU 判别器 (RL 微调)
│   ├── latent_diffusion.py       # 潜在空间高斯扩散
│   ├── noise_schedule.py         # 扩散噪声调度
│   └── property_heads.py         # 属性预测头
├── training/
│   ├── train_vae.py              # VAE MLE 训练 (Phase 1A)
│   ├── train_vae_rl.py           # RL 对抗微调 (Phase 1B)
│   ├── train_diffusion.py        # 扩散训练 (Phase 2)
│   ├── dataset.py                # 数据集与数据加载
│   ├── losses.py                 # 复合损失函数
│   └── utils.py                  # 训练工具 (checkpoint, EMA, logging)
├── generation/
│   ├── unconditional.py          # 无条件生成
│   ├── variant.py                # 条件变体生成 (5 种模式)
│   ├── test_variant.py           # 变体生成测试
│   └── interpolation.py          # 潜在空间插值
├── evaluation/
│   ├── run_evaluation.py         # 评估入口
│   ├── metrics.py                # 评估指标
│   ├── physicochemical.py        # 理化性质分析
│   ├── computational_validation.py # 计算验证
│   └── visualization.py          # 可视化
├── data/
│   ├── prepare_data.py           # 数据预处理
│   └── compute_embeddings.py     # PLM 嵌入预计算
├── checkpoints/                  # 模型权重 (gitignored)
├── data/processed/               # 处理后数据 (gitignored)
└── results/                      # 生成结果 (gitignored)

frontend/                         # 前端展示页面 (React, 独立于核心项目)
```

---

## 七、关键设计决策与技术亮点

1. **多 PLM 后端可切换**: 通过统一接口支持 ESM-2 / Ankh / ProtT5，只需修改配置文件即可切换；嵌入预计算避免训练瓶颈。

2. **BLOSUM62 混合编码**: 将进化信息 (固定) 与任务特定特征 (可学习) 结合，比单纯 one-hot 提供更丰富的残基级表征。

3. **非自回归 Transformer 解码器**: 并行预测所有位置，自注意力提供全局上下文，消除暴露偏差和误差累积；较小的模型规模适配 6.6K 样本的数据集。

4. **Cyclical KL Annealing + Free Bits**: 2 轮周期性退火 (beta_max=1.0) + 每维度最小 KL 下限 (free_bits=0.1)，双重机制防止 posterior collapse。

5. **RL 对抗微调**: Phase 1B 使用 REINFORCE + BiGRU 判别器微调解码器，提升生成序列的真实感。

6. **非自回归变体生成**: 解码器并行输出完整序列后，通过前缀拼接实现 N 端保守 / C 端变异的精确控制。

7. **潜在空间扩散**: 在低维 (64-dim) 潜在空间中运行扩散，相比直接在序列空间扩散大幅降低计算成本，同时通过 CFG 实现属性引导。

8. **多任务训练**: 重建 + KL + 长度预测联合优化（对比学习和属性预测接口保留但当前因数据限制已禁用）。

---

## 八、配置要点

核心配置文件 `esm_diffvae/configs/default.yaml`：

| 参数类别 | 关键参数 | 默认值 |
|---------|---------|--------|
| PLM | backend / model_name | prot_t5 / prot_t5_xl_half |
| VAE | latent_dim / max_seq_len | 64 / 50 |
| 编码 | type | hybrid (BLOSUM62 + learned) |
| 解码器 | hidden_dim / n_layers / n_heads | 128 / 3 / 4 |
| 扩散 | T / schedule / guidance_scale | 50 / cosine / 1.2 |
| VAE 训练 | epochs / lr / beta_max | 300 / 3e-4 / 1.0 |
| RL 微调 | epochs / lr / rl_weight | 50 / 1e-5 / 0.5 |
| 扩散训练 | epochs / lr | 500 / 1e-4 |
| 生成过滤 | min/max_identity | 0.65 / 0.9 |
