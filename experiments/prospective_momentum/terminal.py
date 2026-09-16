"""The launcher's TERMINAL verdict, recorded in the saved JSON (review R1/R2).

Standard library only (no JAX import), so it fits the launcher's bounded
finalization window.

    python -m experiments.prospective_momentum.terminal \
        --run_dir DIR --log_dir DIR --stage STAGE --stage_rc N \
        --integrity_rc N --digest STATE

stage_rc: exit of the stage that ended the computation (0, 3, 4, 124/137 for
the watchdog, anything else a crash; 125 = stage not started for lack of
time).
integrity_rc: 0 verified unchanged; 1 changed or missing; 2 verification not
completed (no time or timed out); 3 no baseline.
digest: "complete", "failed:<why>" or "omitted:<why>".

Verdict rules (worst wins, FAILED > INCOMPLETE > PASS; reasons accumulate,
the stage's own reason first):
  * stage 0 -> PASS, 3 -> INCOMPLETE, 4 -> FAILED, 124/137/125 ->
    INCOMPLETE, other -> FAILED;
  * integrity 1 -> FAILED; 2 or 3 -> never PASS (INCOMPLETE unless worse);
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


def terminal_verdict(stage, stage_rc, integrity_rc, digest, status_exists):
    label, reasons = "PASS", []

    def worsen(new, why):
        nonlocal label
        reasons.append(why)
        if SEVERITY[new] > SEVERITY[label]:
            label = new

    if stage_rc == 0:
        pass
    elif stage_rc == 3:
        worsen("INCOMPLETE", f"{stage} reported INCOMPLETE")
    elif stage_rc == 4:
        worsen("FAILED", f"{stage} reported FAILED")
    elif stage_rc in (124, 137):
        worsen("INCOMPLETE", f"{stage} stopped by the watchdog ({stage_rc})")
    elif stage_rc == 125:
        worsen("INCOMPLETE", f"{stage} not started: no time left")
    else:
        worsen("FAILED", f"{stage} exited with {stage_rc}")

    if integrity_rc == 1:
        worsen("FAILED", "source checksum changed or file missing")
    elif integrity_rc == 2:
        worsen("INCOMPLETE", "source verification not completed; invariance "
                             "not claimed")
    elif integrity_rc == 3:
        worsen("INCOMPLETE", "no source baseline; invariance not claimed")
    elif integrity_rc != 0:
        worsen("FAILED", f"unknown integrity code {integrity_rc}")

    if status_exists and digest != "complete":
        worsen("INCOMPLETE", f"declared digest not produced ({digest}); "
                             "saved results preserved")
    return dict(label=label, code=CODE[label], reasons=reasons, stage=stage,
                stage_rc=stage_rc, integrity_rc=integrity_rc,
                integrity_verified=(integrity_rc == 0), digest=digest)


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
    ap.add_argument("--stage_rc", type=int, required=True)
    ap.add_argument("--integrity_rc", type=int, required=True)
    ap.add_argument("--digest", required=True)
    a = ap.parse_args(argv)
    status_path = os.path.join(a.run_dir, "status.json")
    exists = os.path.isfile(status_path)
    v = terminal_verdict(a.stage, a.stage_rc, a.integrity_rc, a.digest,
                         exists)
    v["recorded_at"] = time.time()
    ok = True
    if exists:
        try:
            with open(status_path) as fh:
                st = json.load(fh)
            st["terminal"] = v
            _atomic_json(status_path, st)
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
