#!/usr/bin/env bash
# One-time: extract Speech Commands 10 MFCC features and write the manifest.
# Run once on the cluster before any training. Idempotent: if the manifest
# verifies, it does nothing.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/cluster_env.sh"
cluster_check_env || { echo "ENVIRONMENT CHECK FAILED"; exit 2; }
cd "$PROSPECTIVE_REPO" || exit 2

DATA_ROOT=${DATA_ROOT:?set DATA_ROOT to the extracted speech_commands_v0.02 dir}
mkdir -p "$PROSPECTIVE_DATA"
rc=0
"$PY" - "$DATA_ROOT" "$PROSPECTIVE_DATA" <<'PYEOF' || rc=$?
import json, sys
from dataloaders import speech_commands10 as SC
root, cache = sys.argv[1], sys.argv[2]
try:
    _, m = SC.load(cache)
    print("cache already present and VERIFIED:", m["counts"])
except Exception as e:
    print("preparing features (", type(e).__name__, ")")
    m = SC.prepare(root, cache)
    print(json.dumps({k: m[k] for k in ("counts", "mfcc", "split_seed",
                                        "file_list_sha256")}, indent=2))
PYEOF
echo "PREPARE_EXIT=$rc  cache=$PROSPECTIVE_DATA"
exit $rc
