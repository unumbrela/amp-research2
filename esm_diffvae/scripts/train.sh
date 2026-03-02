#!/bin/bash
# ESM-DiffVAE End-to-End Training Pipeline
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "============================================"
echo "  ESM-DiffVAE Training Pipeline"
echo "============================================"

# Step 1: Prepare data
echo ""
echo "[Step 1/4] Preparing dataset..."
python data/prepare_data.py --min-len 5 --max-len 50

# Step 2: Compute ESM-2 embeddings
echo ""
echo "[Step 2/4] Computing ESM-2 embeddings..."
python data/compute_embeddings.py --batch-size 16

# Step 3: Train VAE
echo ""
echo "[Step 3/4] Training VAE..."
python training/train_vae.py

# Step 4: Train Diffusion
echo ""
echo "[Step 4/4] Training Latent Diffusion..."
python training/train_diffusion.py

echo ""
echo "============================================"
echo "  Training Complete!"
echo "  Full model: checkpoints/esm_diffvae_full.pt"
echo "============================================"
