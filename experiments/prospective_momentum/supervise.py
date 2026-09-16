"""Deadline supervisor for one launcher stage (review F1, 8a09586).

Standard library only. Replaces GNU `timeout`, which exits as soon as its
DIRECT child is reaped, so a TERM-ignoring descendant of a leader that exits
on TERM would outlive the pending KILL.

    python supervise.py --term_at EPOCH --grace S --outcome FILE -- cmd ...

* The command runs as the leader of a NEW session / process group.
* The supervisor keeps responsibility for that whole group until it is empty:
  - at `term_at` it sends TERM to the group; at `term_at + grace` it sends
    KILL to whatever remains, whether or not the leader has already exited;
  - if the leader exits BEFORE `term_at` but leaves group members behind,
    those are sent TERM, then KILL after `grace` (never later than
    `term_at + grace`), and recorded as orphans cleaned;
  - on Linux it registers as a child subreaper, so orphaned descendants are
    re-parented to it and REAPED here instead of lingering as zombies;
  - if the supervisor itself receives TERM, it moves `term_at` to now and
    performs the same cleanup before exiting.
* Scope: a descendant that deliberately leaves the process group (its own
  setsid/setpgid) is outside this guarantee.

The outcome is written to FILE as one line "OUTCOME RC" and to FILE.json:
  completed          leader exited before term_at; RC = leader exit code
                     (128+signal if killed by a signal)
  watchdog_term      TERM sent at term_at; group empty within grace; RC 124
  watchdog_kill      KILL was needed; group empty afterwards;          RC 137
  supervisor_failure the command could not be started, or members
                     survived KILL;                                   RC 70
If FILE is missing or unparsable the launcher records supervisor_failure.
"""

import argparse
import ctypes
import json
import os
import signal
import subprocess
import sys
import time

POLL_S = 0.05
KILL_CONFIRM_S = 2.0
PR_SET_CHILD_SUBREAPER = 36


def _subreaper():
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        return libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) == 0
    except Exception:
        return False


def _group_alive(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _signal_group(pgid, sig):
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass


def _exitcode(status):
    code = os.waitstatus_to_exitcode(status)
    return 128 - code if code < 0 else code


def supervise(cmd, term_at, grace, now=time.time):
    rec = dict(started=False, outcome="supervisor_failure", rc=70,
               leader_rc=None, term_sent=False, kill_sent=False,
               orphans_cleaned=False, subreaper=_subreaper(), error=None)
    state = dict(term_at=term_at)

    def on_term(signum, frame):
        state["term_at"] = min(state["term_at"], now())
    signal.signal(signal.SIGTERM, on_term)
    try:
        proc = subprocess.Popen(cmd, start_new_session=True)
    except Exception as e:
        rec["error"] = f"could not start: {e!r}"
        return rec
    rec["started"] = True
    pgid = proc.pid
    kill_at = None
    kill_confirm_by = None
    completed_first = False
    while True:
        while True:                              # reap leader and orphans
            try:
                pid, status = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                break
            if pid == 0:
                break
            if pid == proc.pid:
                rec["leader_rc"] = _exitcode(status)
                proc.returncode = rec["leader_rc"]
        t = now()
        alive = _group_alive(pgid)
        if rec["leader_rc"] is not None and not alive:
            break
        if not rec["term_sent"]:
            if t >= state["term_at"]:
                _signal_group(pgid, signal.SIGTERM)
                rec["term_sent"] = True
                kill_at = state["term_at"] + grace
            elif rec["leader_rc"] is not None and alive:
                completed_first = True           # leftovers after completion
                _signal_group(pgid, signal.SIGTERM)
                rec["term_sent"] = True
                kill_at = min(t + grace, state["term_at"] + grace)
        elif kill_at is not None and state["term_at"] + grace < kill_at:
            kill_at = state["term_at"] + grace   # supervisor itself TERMed
        if rec["term_sent"] and not rec["kill_sent"] and t >= kill_at \
                and alive:
            _signal_group(pgid, signal.SIGKILL)
            rec["kill_sent"] = True
            kill_confirm_by = t + KILL_CONFIRM_S
        if rec["kill_sent"] and t > kill_confirm_by:
            rec["error"] = "process-group members survived KILL"
            return rec
        time.sleep(POLL_S)
    if completed_first:
        rec.update(outcome="completed", rc=rec["leader_rc"],
                   orphans_cleaned=True)
    elif not rec["term_sent"]:
        rec.update(outcome="completed", rc=rec["leader_rc"])
    elif rec["kill_sent"]:
        rec.update(outcome="watchdog_kill", rc=137)
    else:
        rec.update(outcome="watchdog_term", rc=124)
    return rec


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--" not in argv:
        print("usage: supervise.py --term_at E --grace S --outcome F -- cmd",
              file=sys.stderr)
        return 70
    i = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--term_at", type=float, required=True)
    ap.add_argument("--grace", type=float, required=True)
    ap.add_argument("--outcome", required=True)
    a = ap.parse_args(argv[:i])
    cmd = argv[i + 1:]
    rec = supervise(cmd, a.term_at, a.grace)
    with open(a.outcome + ".json.partial", "w") as fh:
        json.dump(rec, fh)
    os.replace(a.outcome + ".json.partial", a.outcome + ".json")
    with open(a.outcome + ".partial", "w") as fh:
        fh.write(f"{rec['outcome']} {rec['rc']}\n")
    os.replace(a.outcome + ".partial", a.outcome)
    return 0 if rec["outcome"] != "supervisor_failure" else 70


if __name__ == "__main__":
    sys.exit(main())
