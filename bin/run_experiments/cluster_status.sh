#!/usr/bin/env bash
# Status / resume helper: what exists, what finished, what can continue.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/cluster_env.sh"
cd "$PROSPECTIVE_REPO" 2>/dev/null || true
echo "=== environment ==="; cluster_check_env || true
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
