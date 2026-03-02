#!/bin/bash
# ESM-DiffVAE Generation Examples
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

CHECKPOINT="${1:-checkpoints/esm_diffvae_full.pt}"

echo "============================================"
echo "  ESM-DiffVAE Generation"
echo "  Checkpoint: $CHECKPOINT"
echo "============================================"

# Unconditional generation
echo ""
echo "[1] Unconditional Generation (100 sequences)..."
python generation/unconditional.py \
    --checkpoint "$CHECKPOINT" \
    --n-samples 100 \
    --guidance-scale 2.0 \
    --temperature 1.0 \
    --top-p 0.9

# Variant generation - Magainin-2
echo ""
echo "[2] Variant Generation - Magainin-2..."
python generation/variant.py \
    --checkpoint "$CHECKPOINT" \
    --input-sequence "GIGKFLHSAKKFGKAFVGEIMNS" \
    --n-variants 50 \
    --variation-strength 0.3

# Variant generation - Indolicidin (short AMP)
echo ""
echo "[3] Variant Generation - Indolicidin..."
python generation/variant.py \
    --checkpoint "$CHECKPOINT" \
    --input-sequence "ILPWKWPWWPWRR" \
    --n-variants 50 \
    --variation-strength 0.3

# Interpolation example
echo ""
echo "[4] Latent Space Interpolation (Magainin-2 <-> Indolicidin)..."
python generation/interpolation.py \
    --checkpoint "$CHECKPOINT" \
    --seq-a "GIGKFLHSAKKFGKAFVGEIMNS" \
    --seq-b "ILPWKWPWWPWRR" \
    --n-steps 10

echo ""
echo "============================================"
echo "  Generation Complete!"
echo "  Results in: results/"
echo "============================================"
