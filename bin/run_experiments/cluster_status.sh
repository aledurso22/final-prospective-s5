#!/usr/bin/env bash
# Status / resume helper: what exists, what finished, what can continue.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/cluster_env.sh"
cd "$PROSPECTIVE_REPO" 2>/dev/null || true
echo "=== environment ==="; cluster_check_env || true

echo; echo "=== python packages and backend ==="
"$PY" - <<'PYEOF' 2>&1 | sed 's/^/  /'
import importlib
try:
    import jax
    print("jax        ", jax.__version__, "backend", jax.default_backend())
    print("devices    ", jax.devices())
except Exception as e:
    print("jax        MISSING/BROKEN:", e)
for mod in ("torch", "torchaudio", "flax", "optax", "scipy"):
    try:
        m = importlib.import_module(mod)
        print(f"{mod:<11}", getattr(m, "__version__", "?"))
    except Exception as e:
        print(f"{mod:<11} MISSING ({type(e).__name__})")
PYEOF
echo "  NOTE: torchaudio is required ONCE, on CPU, to extract MFCC features."
echo "        It is not needed for training. Do not upgrade the shared env"
echo "        silently; report a missing package instead."

echo; echo "=== speech commands raw data, likely locations ==="
for d in "${DATA_ROOT:-}" \
         "$PROSPECTIVE_REPO/raw_datasets/speech_commands/0.0.2" \
         "$PROSPECTIVE_RUNS/speech_commands_v0.02" \
         "$HOME/raw_datasets/speech_commands/0.0.2"; do
  [ -n "$d" ] || continue
  if [ -d "$d/yes" ] && [ -d "$d/go" ]; then
    n=$(ls "$d"/yes/*.wav 2>/dev/null | wc -l | tr -d " ")
    echo "  FOUND $d   (yes/ has $n wav files)"
  elif [ -d "$d" ]; then
    echo "  exists but no 10-word subdirs: $d"
  fi
done
echo "  (need a directory containing yes/ no/ up/ down/ left/ right/ on/ off/ stop/ go/)"

echo; echo "=== feature cache ==="
if [ -f "$PROSPECTIVE_DATA/manifest.json" ]; then
  "$PY" -c "import json,sys; m=json.load(open('$PROSPECTIVE_DATA/manifest.json')); print('  counts', m['counts']); print('  split_seed', m['split_seed']); print('  mfcc', m['mfcc'])" 2>&1
else
  echo "  no manifest at $PROSPECTIVE_DATA — run cluster_prepare_data.sh"
fi
echo; echo "=== runs under $PROSPECTIVE_RUNS ==="
for d in "$PROSPECTIVE_RUNS"/*/*/ "$PROSPECTIVE_RUNS"/*/; do
  [ -d "$d" ] || continue
  last=""; best=""
  [ -f "$d/last.msgpack" ] && last="last"
  [ -f "$d/best.msgpack" ] && best="best"
  if [ -f "$d/metrics.jsonl" ]; then
    n=$(wc -l < "$d/metrics.jsonl" | tr -d ' ')
    tailline=$(tail -1 "$d/metrics.jsonl")
    echo "$d  records=$n  ckpt=[$last $best]"
    echo "    $tailline"
  fi
done
echo
echo "To resume an interrupted training run, re-issue the SAME command with"
echo "the same --run_dir: it continues from the last checkpoint."
