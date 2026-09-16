# Sourced by cluster_prospective_momentum.sh and by its terminal-path check.
# Every step is bounded by the ONE absolute DEADLINE (review R2, a1f0439).
#
# Allocation of the 30-second reserve, all measured back from DEADLINE:
#   stage TERM at DEADLINE-30, SIGKILL grace 5 s     -> stage gone by -25
#   source verification (sha256sum -c), until         -> -20
#   digest (read-only JSON), until                    -> -8
#   terminal verdict merge into status.json, until    -> -3
# A step that has no time left is NOT started and is recorded as such.
#
# Descendants: GNU `timeout` (without --foreground) runs the command in its
# own process group and sends TERM, and after the grace KILL, to that whole
# group, so children such as pytest's probe subprocess are included.
#
# Requires: DEADLINE, LOG_DIR, SOURCE_DIR, PY.

PM_GRACE_S=5
PM_VERIFY_END=20
PM_DIGEST_END=8
PM_TERMINAL_END=3
PM_DIGEST_MAX=12
PM_TERMINAL_MAX=5

pm_left() { echo $(( DEADLINE - $(date +%s) )); }

# pm_bounded <seconds until TERM> <grace> cmd... ; 125 = not started
pm_bounded() {
  local lim=$1 grace=$2; shift 2
  [ "$lim" -gt 0 ] || return 125
  timeout --kill-after="${grace}s" --signal=TERM "${lim}s" "$@"
}

# sets PM_INTEGRITY_RC: 0 unchanged, 1 changed/missing, 2 not completed,
# 3 no baseline
pm_verify_source() {
  local base="$LOG_DIR/source_sha256_before.txt" rc lim
  if [ ! -s "$base" ]; then PM_INTEGRITY_RC=3; return 0; fi
  lim=$(( $(pm_left) - PM_VERIFY_END ))
  pm_bounded "$lim" 1 bash -c 'cd "$1" && sha256sum -c "$2"' _ \
    "$SOURCE_DIR" "$base" > "$LOG_DIR/source_sha256_after.txt" 2>&1
  rc=$?
  case "$rc" in
    0) PM_INTEGRITY_RC=0 ;;
    124|125|137) PM_INTEGRITY_RC=2 ;;
    *) PM_INTEGRITY_RC=1 ;;
  esac
  echo "source verification: rc=$rc integrity=$PM_INTEGRITY_RC"
  cat "$LOG_DIR/source_sha256_after.txt" 2>/dev/null || true
}

# pm_digest <run_dir> ; sets PM_DIGEST
pm_digest() {
  local run_dir=$1 rc lim
  if [ ! -f "$run_dir/status.json" ]; then
    PM_DIGEST="omitted:no-status-json"; return 0
  fi
  lim=$(( $(pm_left) - PM_DIGEST_END ))
  [ "$lim" -gt "$PM_DIGEST_MAX" ] && lim=$PM_DIGEST_MAX
  pm_bounded "$lim" 1 "$PY" -u -m experiments.prospective_momentum.summary \
    "$run_dir" > "$LOG_DIR/digest.txt" 2>&1
  rc=$?
  case "$rc" in
    0) PM_DIGEST="complete" ;;
    125) PM_DIGEST="omitted:no-time" ;;
    124|137) PM_DIGEST="failed:timeout" ;;
    *) PM_DIGEST="failed:exit-$rc" ;;
  esac
  echo "digest: $PM_DIGEST ($LOG_DIR/digest.txt)"
  [ "$rc" -eq 125 ] || cat "$LOG_DIR/digest.txt"
}

# pm_terminal <run_dir> <stage> <stage_rc> ; sets PM_LABEL PM_CODE
pm_terminal() {
  local run_dir=$1 stage=$2 stage_rc=$3 lim out rc
  lim=$(( $(pm_left) - PM_TERMINAL_END ))
  [ "$lim" -gt "$PM_TERMINAL_MAX" ] && lim=$PM_TERMINAL_MAX
  out=$(pm_bounded "$lim" 1 "$PY" -u -m experiments.prospective_momentum.terminal \
          --run_dir "$run_dir" --log_dir "$LOG_DIR" --stage "$stage" \
          --stage_rc "$stage_rc" --integrity_rc "$PM_INTEGRITY_RC" \
          --digest "$PM_DIGEST" 2>&1)
  rc=$?
  echo "$out"
  PM_LABEL=$(echo "$out" | sed -n 's/^TERMINAL_VERDICT=\([A-Z]*\) \([0-9]*\)$/\1/p')
  PM_CODE=$(echo "$out" | sed -n 's/^TERMINAL_VERDICT=\([A-Z]*\) \([0-9]*\)$/\2/p')
  if [ "$rc" -ne 0 ] || [ -z "$PM_LABEL" ]; then
    # the verdict could not be computed or persisted: never PASS
    PM_LABEL=FAILED; PM_CODE=4
    echo "terminal verdict not persisted (rc=$rc)"
  fi
}

# pm_finish <run_dir> <stage> <stage_rc> : verify, digest, verdict, exit
pm_finish() {
  local run_dir=$1 stage=$2 stage_rc=$3
  pm_verify_source
  pm_digest "$run_dir"
  pm_terminal "$run_dir" "$stage" "$stage_rc"
  echo "PROSPECTIVE_MOMENTUM_STATUS=$PM_LABEL"
  echo "PROSPECTIVE_MOMENTUM_EXIT=$PM_CODE"
  echo "stage=$stage stage_rc=$stage_rc integrity=$PM_INTEGRITY_RC digest=$PM_DIGEST"
  echo "total elapsed $(( $(date +%s) - START ))s of ${TOTAL_S}s"
  echo "logs=$LOG_DIR"
  exit "$PM_CODE"
}
