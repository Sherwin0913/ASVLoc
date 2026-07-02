#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

GPU_ID="${GPU_ID:-0}"
CONFIG="${CONFIG:-configs/asvloc_default.yaml}"
CHECKPOINT="${CHECKPOINT:-checkpoint/results/final_best_place/asvloc_best_place_recognition.pt}"
RAW_ROOT="${RAW_ROOT:-${REPO_ROOT}/data/ASVInlandRaw}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/eval_asvinland}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python scripts/eval_asvinland_place.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --raw-root "${RAW_ROOT}" \
  --faiss-gpu
