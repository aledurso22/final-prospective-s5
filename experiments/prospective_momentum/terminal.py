"""The launcher's TERMINAL verdict, recorded in the saved JSON (review R1/R2).

Standard library only (no JAX import), so it fits the launcher's bounded
finalization window.

    python -m experiments.prospective_momentum.terminal \
        --run_dir DIR --log_dir DIR --stage STAGE --stage_outcome OUTCOME \
        --stage_rc N --integrity_rc N --digest STATE

stage_outcome (from supervise.py, review F1/F2.3, 8a09586): completed |
watchdog_term | watchdog_kill | not_started | supervisor_failure. "Not
started" is an explicit state, not an exit-code sentinel (GNU timeout's 125
also meant an internal tool failure).
stage_rc: the command's exit code when completed (0, 3, 4; anything else a
crash); ignored otherwise.
integrity_rc: 0 verified unchanged; 1 changed or missing; 2 verification not
completed (no time or watchdog); 3 no baseline; 4 verifier/supervisor
failure.
digest: "complete", "failed:<why>" or "omitted:<why>".

Verdict rules (worst wins, FAILED > INCOMPLETE > PASS; reasons accumulate):
  * the SAVED study verdict, read and validated FIRST when status.json exists
    (review F2): its `study_status` enters with its own severity and its
    `failed`/`incomplete` reasons are kept in the terminal record, so neither
    a watchdog exit nor an outer exit 0 can erase a saved failure or
    incompleteness. An unreadable/invalid status.json -> FAILED; a readable
    one without a finalized study_status -> INCOMPLETE (not finalized);
  * stage: completed 0 -> PASS, 3 -> INCOMPLETE, 4 -> FAILED, other rc ->
    FAILED; watchdog_term / watchdog_kill / not_started -> INCOMPLETE;
    supervisor_failure -> FAILED;
  * integrity 1 or 4 -> FAILED; 2 or 3 -> never PASS (INCOMPLETE unless
    worse);
  * a required digest (a status.json exists) that is not complete -> never
    an unqualified PASS (INCOMPLETE unless worse); completed scientific
    results stay on disk regardless.
The verdict is merged into <run_dir>/status.json under `terminal` when that
file exists, and always written to <log_dir>/terminal.json.
"""

import argparse
import json
import os
import sys
import time

SEVERITY = {"PASS": 0, "INCOMPLETE": 1, "FAILED": 2}
CODE = {"PASS": 0, "INCOMPLETE": 3, "FAILED": 4}
OUTCOMES = ("completed", "watchdog_term", "watchdog_kill", "not_started",
            "supervisor_failure")


def read_saved(status_path):
    """(exists, saved, error). saved is the parsed dict or None."""
    if not os.path.isfile(status_path):
        return False, None, None
    try:
        with open(status_path) as fh:
            st = json.load(fh)
        if not isinstance(st, dict):
            raise ValueError("status.json is not an object")
        return True, st, None
    except Exception as e:
        return True, None, repr(e)


def saved_study_verdict(saved, read_error):
    """The saved study verdict, validated. Returns (label or None, reason,
    detail)."""
    if read_error is not None:
        return "FAILED", f"saved status.json unreadable: {read_error}", None
    label = saved.get("study_status")
    code = saved.get("study_exit")
    detail = dict(study_status=label, study_exit=code,
                  computation_status=saved.get("computation_status"),
                  failed=saved.get("failed"),
                  incomplete=saved.get("incomplete"),
                  integrity_failures=saved.get("integrity_failures"),
                  runtime_failures=saved.get("runtime_failures"))
    if label is None and code is None:
        return ("INCOMPLETE", "saved study status not finalized "
                "(no study_status)", detail)
    if label not in CODE or code != CODE[label]:
        return ("FAILED", f"saved study status invalid: study_status="
                f"{label!r} study_exit={code!r}", detail)
    if label == "PASS":
        return "PASS", None, detail
    why = saved.get("failed") if label == "FAILED" else saved.get("incomplete")
    return label, f"saved study verdict {label}: {why}", detail


def terminal_verdict(stage, stage_outcome, stage_rc, integrity_rc, digest,
                     status_exists, saved=None, read_error=None):
    label, reasons = "PASS", []

    def worsen(new, why):
        nonlocal label
        if why:
            reasons.append(why)
        if SEVERITY[new] > SEVERITY[label]:
            label = new

    saved_detail = None
    if status_exists:
        s_label, s_why, saved_detail = saved_study_verdict(saved, read_error)
        worsen(s_label, s_why)

    if stage_outcome == "completed":
        if stage_rc == 0:
            pass
        elif stage_rc == 3:
            worsen("INCOMPLETE", f"{stage} reported INCOMPLETE")
        elif stage_rc == 4:
            worsen("FAILED", f"{stage} reported FAILED")
        else:
            worsen("FAILED", f"{stage} exited with {stage_rc}")
    elif stage_outcome in ("watchdog_term", "watchdog_kill"):
        worsen("INCOMPLETE", f"{stage} stopped by the watchdog "
                             f"({stage_outcome})")
    elif stage_outcome == "not_started":
        worsen("INCOMPLETE", f"{stage} not started: no time left")
    elif stage_outcome == "supervisor_failure":
        worsen("FAILED", f"{stage}: supervisor failure")
    else:
        worsen("FAILED", f"{stage}: unknown outcome {stage_outcome!r}")

    if integrity_rc == 1:
        worsen("FAILED", "source checksum changed or file missing")
    elif integrity_rc == 2:
        worsen("INCOMPLETE", "source verification not completed; invariance "
                             "not claimed")
    elif integrity_rc == 3:
        worsen("INCOMPLETE", "no source baseline; invariance not claimed")
    elif integrity_rc == 4:
        worsen("FAILED", "source verification tool/supervisor failure")
    elif integrity_rc != 0:
        worsen("FAILED", f"unknown integrity code {integrity_rc}")

    if status_exists and digest != "complete":
        worsen("INCOMPLETE", f"declared digest not produced ({digest}); "
                             "saved results preserved")
    return dict(label=label, code=CODE[label], reasons=reasons, stage=stage,
                stage_outcome=stage_outcome, stage_rc=stage_rc,
                integrity_rc=integrity_rc,
                integrity_verified=(integrity_rc == 0), digest=digest,
                saved_study=saved_detail)


def _atomic_json(path, obj):
    tmp = path + ".partial"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2)
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--log_dir", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--stage_outcome", required=True)
    ap.add_argument("--stage_rc", type=int, required=True)
    ap.add_argument("--integrity_rc", type=int, required=True)
    ap.add_argument("--digest", required=True)
    a = ap.parse_args(argv)
    status_path = os.path.join(a.run_dir, "status.json")
    exists, saved, err = read_saved(status_path)        # read FIRST (F2)
    v = terminal_verdict(a.stage, a.stage_outcome, a.stage_rc,
                         a.integrity_rc, a.digest, exists, saved, err)
    v["recorded_at"] = time.time()
    ok = True
    if exists and saved is not None:
        try:
            saved["terminal"] = v
            _atomic_json(status_path, saved)
        except Exception as e:                     # persistence failure
            v = dict(v, label="FAILED", code=4,
                     reasons=v["reasons"] + [f"status.json not updated: {e!r}"])
            ok = False
    _atomic_json(os.path.join(a.log_dir, "terminal.json"), v)
    print(f"TERMINAL_VERDICT={v['label']} {v['code']}")
    for r in v["reasons"]:
        print(f"TERMINAL_REASON={r}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
