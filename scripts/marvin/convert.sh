#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../robodojo/common.sh"
exec "$python_bin" -m vpp2.cli marvin-convert "$@"
