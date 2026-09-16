# Extra integrity verification for the independent-source replication:
# the NEW source checkpoints created inside the run directory.
#
# Sourced by cluster_prospective_momentum_replication.sh and by its focused
# check. `pm_finish` (prospective_momentum_terminal.sh) calls `pm_extra_verify`
# when it is defined; it is undefined for every earlier study, which therefore
# behaves exactly as before.
#
# It only WORSENS PM_INTEGRITY_RC (0 verified, 1 changed/missing file,
# 2 not completed, 3 no baseline, 4 verifier/supervisor failure), so a failure
# already found on the completed study's read-only source cannot be erased.

pm_worst_integrity() {           # severity: 0 < 2 = 3 < 1 = 4
  local a=$1 b=$2
  local sa sb
  case "$a" in 0) sa=0 ;; 2|3) sa=1 ;; *) sa=2 ;; esac
  case "$b" in 0) sb=0 ;; 2|3) sb=1 ;; *) sb=2 ;; esac
  if [ "$sb" -gt "$sa" ]; then echo "$b"; else echo "$a"; fi
}

# pm_extra_verify <run_dir> : verifies <run_dir>/sources/SHA256SUMS
pm_extra_verify() {
  local run_dir=$1 base="$1/sources/SHA256SUMS" rc
  if [ ! -s "$base" ]; then
    echo "new sources: no SHA256SUMS baseline in $run_dir/sources"
    PM_INTEGRITY_RC=$(pm_worst_integrity "$PM_INTEGRITY_RC" 3)
    return 0
  fi
  pm_bounded $(( DEADLINE - PM_VERIFY_END )) 1 \
    bash -c 'cd "$1/sources" && sha256sum -c SHA256SUMS' _ "$run_dir" \
    > "$LOG_DIR/new_sources_sha256_after.txt" 2>&1
  case "$PM_OUTCOME" in
    completed) if [ "$PM_RC" = "0" ]; then rc=0; else rc=1; fi ;;
    watchdog_term|watchdog_kill|not_started) rc=2 ;;
    *) rc=4 ;;
  esac
  echo "new sources verification: outcome=$PM_OUTCOME rc=$PM_RC integrity=$rc"
  cat "$LOG_DIR/new_sources_sha256_after.txt" 2>/dev/null || true
  PM_INTEGRITY_RC=$(pm_worst_integrity "$PM_INTEGRITY_RC" "$rc")
}
