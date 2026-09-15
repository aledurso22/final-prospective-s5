#!/usr/bin/env bash
# One-time download of Speech Commands v0.02 (~2.3 GB) and structural check.
#
# The archive extracts FLAT (no top-level directory), so it is always unpacked
# into a dedicated directory. Idempotent: if the ten word directories are
# already present, nothing is downloaded.
#
# No checksum is asserted here because none is published alongside the archive
# that we can verify against. Instead the extracted tree is checked
# STRUCTURALLY, and the feature manifest later records a SHA-256 of the exact
# per-split file list, which is what makes a run auditable.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/cluster_env.sh"

URL=${URL:-http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz}
DEST=${DEST:-/Local/durso/speech_commands_v0.02}
WORDS="yes no up down left right on off stop go"

mkdir -p "$DEST" || { echo "cannot create $DEST"; exit 2; }

missing=0
for w in $WORDS; do [ -d "$DEST/$w" ] || missing=1; done
if [ "$missing" -eq 0 ]; then
  echo "already extracted at $DEST"
else
  TAR="$DEST/speech_commands_v0.02.tar.gz"
  if [ ! -s "$TAR" ]; then
    echo "downloading $URL"
    curl -fL --retry 3 -o "$TAR" "$URL" || { echo "DOWNLOAD FAILED"; exit 2; }
  else
    echo "archive already present: $TAR"
  fi
  echo "extracting into $DEST"
  tar -xzf "$TAR" -C "$DEST" || { echo "EXTRACT FAILED"; exit 2; }
fi

echo "=== structural check ==="
rc=0
total=0
for w in $WORDS; do
  if [ -d "$DEST/$w" ]; then
    n=$(find "$DEST/$w" -name '*.wav' | wc -l | tr -d ' ')
    total=$(( total + n ))
    printf '  %-6s %6s wav\n' "$w" "$n"
    [ "$n" -gt 0 ] || rc=1
  else
    echo "  $w  MISSING"; rc=1
  fi
done
echo "  total  $total wav across 10 words"
df -h "$DEST" | tail -1 | sed 's/^/  disk: /'
# The archive is no longer needed once extracted. /Local is shared and was
# observed at 100% use, so reclaim the 2.3 GB rather than leave it lying about.
if [ "$rc" -eq 0 ] && [ -s "$DEST/speech_commands_v0.02.tar.gz" ]; then
  echo "  removing archive to reclaim space: $DEST/speech_commands_v0.02.tar.gz"
  rm -f "$DEST/speech_commands_v0.02.tar.gz"
  df -h "$DEST" | tail -1 | sed 's/^/  disk after: /'
fi
echo "DATA_ROOT=$DEST"
[ "$rc" -eq 0 ] && echo "FETCH_OK" || echo "FETCH_INCOMPLETE"
exit $rc
