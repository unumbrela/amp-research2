# AMP Forge

通用抗菌肽（AMP）生成项目，聚焦“可用、可扩展、可验证”的序列设计工作流。

- 前端展示页面: https://unumbrela.github.io/amp-research2/

## 项目亮点

- 无条件生成（de novo generation）
  - 直接从先验分布采样，批量生成全新 AMP 候选序列。
- 条件变体生成（variant generation）
  - 以已有肽序列为母体，生成结构上可控、功能导向的变体集合。
- 统一模型框架
  - 基于 ESM 表征 + VAE + Latent Diffusion，兼顾表达能力与可控性。
- 工程化落地
  - 提供训练、生成、评估脚本，可直接用于实验迭代。

## 仓库结构

- `esm_diffvae/`: 通用抗菌肽生成核心项目（重点）
- `frontend/`: 前端可视化与静态站点工程（配套展示）

## 快速开始（esm_diffvae）

```bash
cd esm_diffvae
pip install -r requirements.txt
```

### 1) 无条件生成 AMP 序列

```bash
cd esm_diffvae
python generation/unconditional.py \
  --checkpoint checkpoints/esm_diffvae_full.pt \
  --n-samples 100 \
  --top-p 0.9
```

### 2) 基于母序列生成变体

```bash
cd esm_diffvae
python generation/variant.py \
  --checkpoint checkpoints/esm_diffvae_full.pt \
  --input-sequence "GIGKFLHSAKKFGKAFVGEIMNS" \
  --mode mixed \
  --n-variants 50
```

可选模式：`mixed`、`c_sub`、`c_ext`、`c_trunc`、`tag`、`latent`。

## 训练与评估

```bash
cd esm_diffvae
bash scripts/train.sh
bash scripts/evaluate.sh
```

## 前端（简述）

前端用于展示 AMP 研究内容与结果，不影响 `esm_diffvae` 核心生成流程。

本地启动：

```bash
cd frontend
pnpm install
pnpm dev
```

## 说明

- 仓库已配置忽略训练权重、数据缓存与生成结果等大文件目录（如 `esm_diffvae/checkpoints/`、`esm_diffvae/data/processed/`、`esm_diffvae/results/`）。
- 如需共享模型权重，建议使用 Release 或外部对象存储链接。
