#!/usr/bin/env bash
set -eo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MARVIN_ENV="${MARVIN_GR00T_ENV:-/home/nvidia/marvin_gr00t_runtime/env.sh}"
if [[ ! -r "${MARVIN_ENV}" ]]; then
    echo "[marvin-vpp2][ERROR] environment file not found: ${MARVIN_ENV}" >&2
    exit 2
fi
source "${MARVIN_ENV}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec python3 "${PROJECT_ROOT}/run_vpp2_client.py" "$@"
