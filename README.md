# AMP Research 2026

抗菌肽（AMP）生成技术的可视化调研项目，包含模型演进、评估方法、开源仓库与实现建议。

## 在线访问

- GitHub Pages: https://unumbrela.github.io/amp-research2/

## 本地运行

```bash
pnpm install
pnpm dev
```

默认使用 Vite 本地开发服务（通常为 `http://localhost:3000` 或自动分配端口）。

## 构建与预览

```bash
pnpm build
pnpm preview
```

## 项目结构

- `client/`: React + Vite 前端代码
- `server/`: 静态托管入口（生产运行）
- `shared/`: 共享常量
- `patches/`: `pnpm` 依赖补丁

## 部署

仓库包含 GitHub Actions 工作流 `.github/workflows/deploy-pages.yml`。推送到 `main` 分支后会自动部署到 GitHub Pages。
