#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/cluster_env.sh"
cd "$PROSPECTIVE_REPO"

DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to extracted speech_commands_v0.02}"
DATA_CACHE="${S5_THREE_ARM_DATA:-/Users/durso/s5-runs/sc10_official_cache}"
mkdir -p "$DATA_CACHE"
"$PY" - "$DATA_ROOT" "$DATA_CACHE" <<'PYEOF'
import json
import sys
from experiments.s5_three_arm_full.data import (prepare_official_raw,
                                                 validate_official_raw_cache)

manifest = prepare_official_raw(sys.argv[1], sys.argv[2])
validate_official_raw_cache(sys.argv[2], ("train", "val", "test"))
print(json.dumps({"split_definition": manifest["split_definition"],
                  "counts": manifest["counts"],
                  "representation": manifest["representation"]}, indent=2))
PYEOF
