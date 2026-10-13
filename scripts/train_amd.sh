#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    printf 'Error: .venv is missing. Run scripts/install_amd.sh first.\n' >&2
    exit 1
fi

# RX 6700 XT / gfx1031 uses this community compatibility override on this setup.
# This is not officially supported by AMD; callers may override both variables.
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-10.3.0}"
export MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE:-FAST}"

exec "$PYTHON" "$ROOT/train.py" "$@"
