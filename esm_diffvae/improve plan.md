
 当前 VAE 阶段（Phase 1）的两个解码器版本均表现不佳：
 - v6 (自回归 GRU): train_acc=94.5%, val_acc=55.1% — 曝光偏差导致 40% 差距
 - v7 (非自回归 Conv): train_acc=64.5%, val_acc=44% — 缺乏位置间全局上下文

 参考项目分析：
 - diff-amp: GAN + DIFFormer 注意力 + 三阶段训练 (MLE → 判别器 → 对抗/RL)，模型极小 (embed=3, hidden=128)
 - AMPainter: 微调 Ankh (T5) + RL 智能体 (REINFORCE) 学习突变位置，HyperAMP 评分

 根因诊断：
 1. 编码器瓶颈: 1060维(ProtT5 1024 + BLOSUM62 36) 经单层线性压缩至 256维，信息大量丢失
 2. Conv 解码器无全局上下文: kernel=5 只看 ±2 邻居，无法建模位置间依赖
 3. 损失函数配置不当: 对比损失过强(λ=0.2, 实际贡献 4.04 > 重建损失)、属性预测在空数据上训练、KL 太弱(β_max=0.25)
 4. max_seq_len=50 但数据最长 30: 白白浪费 20 个位置的计算和学习容量

 修改方案概览

 Phase 1A: VAE 预训练 — Transformer 解码器 + 简化损失 + 编码器修复
 Phase 1B: RL 精调 — 添加判别器 + REINFORCE 策略梯度优化生成质量
 Phase 2: Diffusion — 无结构改动，在新编码器的 latent z 上重新训练

 ---
 Step 1: 解码器重写 — 非自回归 Transformer

 文件: esm_diffvae/models/decoder.py (完全重写)

 用 Transformer self-attention 替换 Conv blocks：

 z + properties → cond_proj (2层MLP) → [B, hidden_dim]
                               ↓ broadcast 到所有位置
             + learnable position embeddings [max_len, hidden_dim]
                               ↓
             TransformerBlock × 3 (pre-norm self-attention + FFN)
                               ↓
             output_proj → [B, max_len, vocab_size=21]

 关键参数（匹配 6.6K 小数据集）：
 - hidden_dim = 128（从 256 缩小，防止过拟合）
 - n_heads = 4（head_dim = 32）
 - n_layers = 3
 - ffn_dim = 256（2× hidden_dim）
 - dropout = 0.3
 - Pre-LayerNorm（训练更稳定）
 - 约 350K 参数（比 Conv 解码器的 1.2M 小很多）

 为什么能解决问题：
 - 非自回归 → 无曝光偏差（同 Conv）
 - Self-attention → 每个位置看到所有其他位置（解决 Conv 的局部窗口问题）
 - 更小模型 → 6.6K 数据集不容易过拟合

 保持 API 兼容: forward(z, properties, target_indices=None, teacher_forcing_ratio=1.0, target_len=None) → (logits,
 length_logits)

 Step 2: 编码器修复 — 多层渐进压缩

 文件: esm_diffvae/models/encoder.py (修改 input_proj)

 # 当前（瓶颈）:
 self.input_proj = nn.Sequential(
     nn.Linear(1060, 256),  # 一步 4× 压缩
     nn.LayerNorm(256), nn.GELU()
 )

 # 改为（渐进压缩）:
 self.input_proj = nn.Sequential(
     nn.Linear(1060, 512),  # 1060 → 512 (~2×)
     nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
     nn.Linear(512, 256),   # 512 → 256 (~2×)
     nn.LayerNorm(256), nn.GELU()
 )

 增加约 270K 参数，但提供更平滑的信息压缩路径。

 Step 3: 损失函数简化

 文件: esm_diffvae/training/losses.py (修改 ESMDiffVAELoss)

 ┌────────────────┬─────────────────────────────────┬────────────────────────────────┬─────────────────────────────┐
 │     损失项     │              当前               │              改为              │            原因             │
 ├────────────────┼─────────────────────────────────┼────────────────────────────────┼─────────────────────────────┤
 │ Reconstruction │ CE, label_smoothing=0.05        │ CE, label_smoothing=0.02       │ 21 类词表 0.05 太大         │
 ├────────────────┼─────────────────────────────────┼────────────────────────────────┼─────────────────────────────┤
 │ KL             │ β_max=0.25, warmup=200, 4       │ β_max=1.0, warmup=100, 2       │ 增强 latent 正则化          │
 │                │ cycles                          │ cycles                         │                             │
 ├────────────────┼─────────────────────────────────┼────────────────────────────────┼─────────────────────────────┤
 │ Free bits      │ 0.05                            │ 0.1                            │ 防止 posterior collapse     │
 ├────────────────┼─────────────────────────────────┼────────────────────────────────┼─────────────────────────────┤
 │ Contrastive    │ λ=0.2                           │ λ=0.0 (移除)                   │ 单标签 SupCon               │
 │                │                                 │                                │ 过强，干扰重建              │
 ├────────────────┼─────────────────────────────────┼────────────────────────────────┼─────────────────────────────┤
 │ Property       │ λ=0.2                           │ λ=0.0 (移除)                   │ mic/toxic/hemolytic 全空    │
 ├────────────────┼─────────────────────────────────┼────────────────────────────────┼─────────────────────────────┤
 │ Length         │ λ=0.2                           │ λ=0.2 (保持)                   │ 正常工作                    │
 └────────────────┴─────────────────────────────────┴────────────────────────────────┴─────────────────────────────┘

 简化后: total = recon + β·KL + 0.2·length_loss

 在 ESMDiffVAELoss.forward() 中，当 lambda_contrastive=0 或 lambda_property=0 时跳过计算。

 Step 4: 配置更新

 文件: esm_diffvae/configs/default.yaml

 关键改动：
 vae:
   max_seq_len: 30           # 50→30, 匹配数据集实际最大长度
   n_decoder_layers: 3       # Transformer 层数
   decoder_hidden_dim: 128   # 解码器隐藏维度(从256缩小)
   decoder_n_heads: 4        # 新增: 注意力头数
   decoder_ffn_dim: 256      # 新增: FFN 中间维度
   dropout: 0.3

 train_vae:
   epochs: 300
   beta_max: 1.0             # 0.25→1.0
   beta_warmup_epochs: 100   # 200→100
   kl_n_cycles: 2            # 4→2
   free_bits: 0.1            # 0.05→0.1
   lambda_contrastive: 0.0   # 移除
   lambda_property: 0.0      # 移除
   label_smoothing: 0.02     # 0.05→0.02
   early_stopping_patience: 40
   teacher_forcing_start: 1.0  # 兼容保留(non-AR 不使用)
   teacher_forcing_end: 1.0
   teacher_forcing_warmup: 0

 # 新增 RL 精调配置
 train_vae_rl:
   epochs: 50
   lr: 1.0e-5
   disc_lr: 1.0e-4
   rl_weight: 0.5
   disc_steps_per_gen_step: 3
   temperature: 0.8
   disc_embed_dim: 32
   disc_hidden_dim: 64

 Step 5: 主模型类适配

 文件: esm_diffvae/models/esm_diffvae.py

 - 传递新配置键 (decoder_n_heads, decoder_ffn_dim) 给解码器构造函数
 - decode() API 不变，生成脚本无需修改

 Step 6: 判别器模型（新文件）

 新文件: esm_diffvae/models/discriminator.py

 class SequenceDiscriminator(nn.Module):
     """BiGRU 判别器：区分真实序列与生成序列"""
     # Embedding(21, 32) → BiGRU(32, 64, 1层) → Linear(128, 1) → Sigmoid
     # 约 30K 参数 — 必须远小于生成器以避免模式崩溃

 参考 diff-amp 的判别器设计（BiGRU + FC，但更小）。

 Step 7: RL 精调训练脚本（新文件）

 新文件: esm_diffvae/training/train_vae_rl.py

 Phase 1B 训练流程：
 1. 加载 Phase 1A 预训练的 VAE checkpoint
 2. 冻结编码器，只训练解码器 + 判别器
 3. 每个 epoch：
   - 判别器步骤 (×3): 编码真实数据得 z → 解码 → argmax 采样序列，与真实序列混合，BCE 损失训练判别器
   - 生成器步骤 (×1): 解码得 logits → Gumbel-softmax 采样 → 判别器评分作为 reward → REINFORCE 梯度 + MLE 重建损失混合
 4. loss = 0.5 * recon_loss + 0.5 * rl_loss（MLE 防止灾难性遗忘）

 # REINFORCE 核心:
 probs = F.softmax(logits / temperature, dim=-1)
 sampled = torch.multinomial(probs.view(-1, V), 1).view(B, L)
 reward = discriminator(sampled)  # [B]
 baseline = running_mean_reward
 log_probs = F.log_softmax(logits, dim=-1).gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
 rl_loss = -((reward - baseline) * log_probs.sum(dim=1)).mean()

 Step 8: 训练脚本更新

 文件: esm_diffvae/scripts/train.sh

 更新流水线：
 # Step 1-2: 数据准备 + PLM 嵌入（不变）
 # Step 3: Phase 1A — VAE MLE 预训练
 python training/train_vae.py --config configs/default.yaml
 # Step 4: Phase 1B — RL 精调（新增）
 python training/train_vae_rl.py --config configs/default.yaml
 # Step 5: Phase 2 — Latent Diffusion（不变）
 python training/train_diffusion.py --config configs/default.yaml

 Phase 2 Diffusion 兼容性

 无需结构改动。Latent Diffusion 在编码器的 z ∈ R^64 上运行，latent_dim 保持 64，编码器输出格式 (mu, logvar)
 不变。train_diffusion.py 已有逻辑会重新编码训练数据生成 latent vectors。

 唯一注意：max_seq_len 从 50 改为 30，需确保 train_diffusion.py 中的数据加载使用新值（它读取 config，自动适配）。

 ---
 参数预算

 ┌──────────┬──────────────┬────────────────────────┐
 │   组件   │  v7 (当前)   │        v8 (新)         │
 ├──────────┼──────────────┼────────────────────────┤
 │ 编码器   │ ~1.1M        │ ~1.4M (+270K 多层投影) │
 ├──────────┼──────────────┼────────────────────────┤
 │ 解码器   │ ~1.2M (Conv) │ ~350K (Transformer)    │
 ├──────────┼──────────────┼────────────────────────┤
 │ 判别器   │ —            │ ~30K (Phase 1B)        │
 ├──────────┼──────────────┼────────────────────────┤
 │ 总可训练 │ ~2.3M        │ ~1.8M                  │
 └──────────┴──────────────┴────────────────────────┘

 预期效果

 ┌───────────────┬──────────┬───────────┬───────────┐
 │     指标      │ v6 (GRU) │ v7 (Conv) │ v8 (目标) │
 ├───────────────┼──────────┼───────────┼───────────┤
 │ Train Acc     │ 94.5%    │ 64.5%     │ ≥85%      │
 ├───────────────┼──────────┼───────────┼───────────┤
 │ Val Acc       │ 55.1%    │ 44%       │ ≥75%      │
 ├───────────────┼──────────┼───────────┼───────────┤
 │ Train-Val Gap │ 39.4%    │ 20.5%     │ ≤10%      │
 ├───────────────┼──────────┼───────────┼───────────┤
 │ Val PPL       │ 5.86     │ ~6.8      │ ≤3.0      │
 └───────────────┴──────────┴───────────┴───────────┘

 验证方式

 1. 模型构建测试: 构建模型，检查 forward pass shape 和参数数量
 2. Phase 1A 训练: 运行 python training/train_vae.py，观察 train/val acc 差距是否 ≤10%
 3. Phase 1B 训练: 运行 python training/train_vae_rl.py，观察判别器是否稳定在 ~55% 准确率
 4. Phase 2 兼容: 运行 python training/train_diffusion.py，确认 latent 编码和扩散训练正常
 5. 生成测试: 运行 python generation/unconditional.py 和 python generation/variant.py，检查生成序列质量

 实施顺序

 1. models/decoder.py — Transformer 解码器（核心改动）
 2. models/encoder.py — 多层输入投影
 3. training/losses.py — 简化损失函数
 4. configs/default.yaml — 超参数更新
 5. models/esm_diffvae.py — 适配新解码器配置
 6. models/discriminator.py — 新文件：判别器
 7. training/train_vae_rl.py — 新文件：RL 精调脚本
 8. scripts/train.sh — 更新训练流水线
 9. 验证 + 测试