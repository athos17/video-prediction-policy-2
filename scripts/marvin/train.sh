#!/usr/bin/env bash
# One launch on each GPU-bearing role, sharing paths and output_dir.
source "$(dirname "${BASH_SOURCE[0]}")/../robodojo/common.sh"
if [[ "${1:-}" == --help || "${1:-}" == help ]]; then
  echo 'Usage: bash scripts/marvin/train.sh [--dry-run] [key=value ...]'
  echo 'Marvin joint Video + Action1B.'
  exit 0
fi
export NNODES="${NNODES:-4}" NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export VPP2_TRAIN_MODULE=vpp2.cli
export VPP2_ACCELERATE_CONFIG=configs/marvin/accelerate_zero2.yaml
exec bash scripts/train.sh --config configs/marvin/train_joint.yaml "$@"
