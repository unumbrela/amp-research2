#!/bin/bash
# ESM-DiffVAE Full Evaluation
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

CHECKPOINT="${1:-checkpoints/esm_diffvae_full.pt}"

echo "============================================"
echo "  ESM-DiffVAE Evaluation"
echo "  Checkpoint: $CHECKPOINT"
echo "============================================"

python evaluation/run_evaluation.py --checkpoint "$CHECKPOINT"

echo ""
echo "Results saved to results/evaluation/"
