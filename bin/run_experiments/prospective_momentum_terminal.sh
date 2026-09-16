# Sourced by cluster_prospective_momentum.sh and by its terminal-path check.
# Every step is bounded by the ONE absolute DEADLINE (review R2, a1f0439).
#
# Allocation of the 30-second reserve, all measured back from DEADLINE:
#   stage TERM at DEADLINE-30, KILL of the whole group 5 s later -> gone by -25
#   source verification (sha256sum -c), TERM by                 -> -20 (+1 s)
#   digest (read-only JSON), TERM by                             -> -8 (+1 s)
#   terminal verdict merge into status.json, TERM by             -> -3 (+1 s)
# A step that has no time left is NOT started: PM_OUTCOME=not_started, an
# explicit state, not an exit-code sentinel (review F2.3, 8a09586).
#
# Descendants (review F1, 8a09586): each step runs under
# experiments/prospective_momentum/supervise.py, which starts the command as
# the leader of its own process group and keeps responsibility for the WHOLE
# group until it is empty - TERM at the step's time, KILL to any remaining
# member after the grace even if the leader has already exited, cleanup of
# members left behind by a leader that completed, and (Linux) reaping of
# orphaned descendants as a child subreaper. GNU `timeout` is no longer used:
# it stops supervising once its direct child is reaped.
#
# Requires: DEADLINE, LOG_DIR, SOURCE_DIR, PY. Optional: PM_SUPERVISOR,
# PM_DIGEST_MODULE (digest entry point), pm_extra_verify (extra integrity).

PM_GRACE_S=5
PM_VERIFY_END=20
PM_DIGEST_END=8
PM_TERMINAL_END=3
PM_DIGEST_MAX=12
PM_TERMINAL_MAX=5
PM_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${PM_SUPERVISOR:=$PM_LIB_DIR/../../experiments/prospective_momentum/supervise.py}"

pm_left() { echo $(( DEADLINE - $(date +%s) )); }

# pm_bounded <term_at epoch> <grace s> cmd...
# Sets PM_OUTCOME in {completed, watchdog_term, watchdog_kill, not_started,
# supervisor_failure} and PM_RC (the command's exit code when completed).
# Always returns 0; callers read PM_OUTCOME and PM_RC.
pm_bounded() {
  local term_at=$1 grace=$2 ofile line
  shift 2
  PM_RC=""
  if [ $(( term_at - $(date +%s) )) -le 0 ]; then
    PM_OUTCOME=not_started
    return 0
  fi
  ofile="$LOG_DIR/.supervise.$$.$RANDOM"
  rm -f "$ofile" "$ofile.json"
  "$PY" "$PM_SUPERVISOR" --term_at "$term_at" --grace "$grace" \
    --outcome "$ofile" -- "$@"
  line=$(cat "$ofile" 2>/dev/null || true)
  PM_OUTCOME=${line%% *}
  PM_RC=${line#* }
  case "$PM_OUTCOME" in
    completed|watchdog_term|watchdog_kill|supervisor_failure) ;;
    *) PM_OUTCOME=supervisor_failure; PM_RC="" ;;
  esac
  [ -f "$ofile.json" ] && cp "$ofile.json" "$LOG_DIR/supervise_last.json"
  rm -f "$ofile"
  return 0
}

# sets PM_INTEGRITY_RC: 0 unchanged, 1 changed/missing, 2 not completed
# (no time or watchdog), 3 no baseline, 4 verifier/supervisor failure
pm_verify_source() {
  local base="$LOG_DIR/source_sha256_before.txt"
  if [ ! -s "$base" ]; then PM_INTEGRITY_RC=3; return 0; fi
  pm_bounded $(( DEADLINE - PM_VERIFY_END )) 1 \
    bash -c 'cd "$1" && sha256sum -c "$2"' _ "$SOURCE_DIR" "$base" \
    > "$LOG_DIR/source_sha256_after.txt" 2>&1
  case "$PM_OUTCOME" in
    completed) if [ "$PM_RC" = "0" ]; then PM_INTEGRITY_RC=0
               else PM_INTEGRITY_RC=1; fi ;;
    watchdog_term|watchdog_kill|not_started) PM_INTEGRITY_RC=2 ;;
    *) PM_INTEGRITY_RC=4 ;;
  esac
  echo "source verification: outcome=$PM_OUTCOME rc=$PM_RC integrity=$PM_INTEGRITY_RC"
  cat "$LOG_DIR/source_sha256_after.txt" 2>/dev/null || true
}

# pm_digest <run_dir> ; sets PM_DIGEST
pm_digest() {
  local run_dir=$1 term_at
  if [ ! -f "$run_dir/status.json" ]; then
    PM_DIGEST="omitted:no-status-json"; return 0
  fi
  term_at=$(( DEADLINE - PM_DIGEST_END ))
  [ $(( term_at - $(date +%s) )) -gt "$PM_DIGEST_MAX" ] && \
    term_at=$(( $(date +%s) + PM_DIGEST_MAX ))
  pm_bounded "$term_at" 1 "$PY" -u -m "${PM_DIGEST_MODULE:-experiments.prospective_momentum.summary}" \
    "$run_dir" > "$LOG_DIR/digest.txt" 2>&1
  case "$PM_OUTCOME" in
    completed) if [ "$PM_RC" = "0" ]; then PM_DIGEST="complete"
               else PM_DIGEST="failed:exit-$PM_RC"; fi ;;
    not_started) PM_DIGEST="omitted:no-time" ;;
    watchdog_term|watchdog_kill) PM_DIGEST="failed:timeout" ;;
    *) PM_DIGEST="failed:supervisor" ;;
  esac
  echo "digest: $PM_DIGEST ($LOG_DIR/digest.txt)"
  [ "$PM_OUTCOME" = not_started ] || cat "$LOG_DIR/digest.txt"
}

# pm_terminal <run_dir> <stage> <stage_outcome> <stage_rc> ; sets PM_LABEL PM_CODE
pm_terminal() {
  local run_dir=$1 stage=$2 s_outcome=$3 s_rc=$4 term_at tfile
  term_at=$(( DEADLINE - PM_TERMINAL_END ))
  [ $(( term_at - $(date +%s) )) -gt "$PM_TERMINAL_MAX" ] && \
    term_at=$(( $(date +%s) + PM_TERMINAL_MAX ))
  tfile="$LOG_DIR/terminal_stdout.txt"
  pm_bounded "$term_at" 1 "$PY" -u -m experiments.prospective_momentum.terminal \
    --run_dir "$run_dir" --log_dir "$LOG_DIR" --stage "$stage" \
    --stage_outcome "$s_outcome" --stage_rc "${s_rc:--1}" \
    --integrity_rc "$PM_INTEGRITY_RC" --digest "$PM_DIGEST" > "$tfile" 2>&1
  cat "$tfile"
  PM_LABEL=$(sed -n 's/^TERMINAL_VERDICT=\([A-Z]*\) \([0-9]*\)$/\1/p' "$tfile")
  PM_CODE=$(sed -n 's/^TERMINAL_VERDICT=\([A-Z]*\) \([0-9]*\)$/\2/p' "$tfile")
  if [ "$PM_OUTCOME" != completed ] || [ "$PM_RC" != "0" ] || [ -z "$PM_LABEL" ]; then
    # the verdict could not be computed or persisted: never PASS
    PM_LABEL=FAILED; PM_CODE=4
    echo "terminal verdict not persisted (outcome=$PM_OUTCOME rc=$PM_RC)"
  fi
}

# pm_finish <run_dir> <stage> <stage_outcome> <stage_rc> : verify, digest,
# verdict, exit
pm_finish() {
  local run_dir=$1 stage=$2 s_outcome=$3 s_rc=$4
  pm_verify_source
  # a study may add its own integrity verification (it can only WORSEN
  # PM_INTEGRITY_RC); undefined for every earlier study
  if declare -F pm_extra_verify > /dev/null; then pm_extra_verify "$run_dir"; fi
  pm_digest "$run_dir"
  pm_terminal "$run_dir" "$stage" "$s_outcome" "$s_rc"
  echo "PROSPECTIVE_MOMENTUM_STATUS=$PM_LABEL"
  echo "PROSPECTIVE_MOMENTUM_EXIT=$PM_CODE"
  echo "stage=$stage outcome=$s_outcome rc=$s_rc integrity=$PM_INTEGRITY_RC digest=$PM_DIGEST"
  echo "total elapsed $(( $(date +%s) - START ))s of ${TOTAL_S}s"
  echo "logs=$LOG_DIR"
  exit "$PM_CODE"
}
