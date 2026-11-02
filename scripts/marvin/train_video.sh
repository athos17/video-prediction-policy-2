#!/usr/bin/env bash
# Run on every GPU-bearing role, sharing data and output_dir.
source "$(dirname "${BASH_SOURCE[0]}")/../robodojo/common.sh"
if [[ "${1:-}" == --help || "${1:-}" == help ]]; then
  echo 'Usage: bash scripts/marvin/train_video.sh [--dry-run] [key=value ...]'
  echo 'Stage-2 Video -> Marvin history-conditioned Video; global batch 128.'
  exit 0
fi
export NNODES="${NNODES:-4}" NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export VPP2_TRAIN_MODULE=vpp2.cli
export VPP2_ACCELERATE_CONFIG=configs/robodojo/accelerate.yaml
exec bash scripts/train.sh --config configs/marvin/train_video.yaml "$@"
