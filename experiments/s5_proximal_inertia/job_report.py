"""Cluster-job bookkeeping for the proximal inertial run: P-audit status and summary.

    python -m experiments.s5_proximal_inertia.job_report p-audit-status <block_audit.json>
        prints "PASS audited=N unavailable=0", "PARTIAL audited=N unavailable=M",
        or "FAIL <reason>" (exit status 1 for FAIL)
    python -m experiments.s5_proximal_inertia.job_report summary <run root>
        prints the end-of-job table; missing or null telemetry is shown as n/a
        and never raises (reporting must not change a training job's outcome)
"""
import json
import math
from pathlib import Path
import sys

ARMS = ("proximal_inertial_s5", "proximal_inertial_control_s5")
SEEDS = (301, 302)


def p_audit_status(report_path):
    try:
        report = json.loads(Path(report_path).read_text())
        n, u = len(report["tasks"]), len(report["unavailable"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return f"FAIL unreadable_report: {type(exc).__name__}"
    if n == 0:
        return f"FAIL audited=0 unavailable={u}"
    return f"{'PASS' if u == 0 else 'PARTIAL'} audited={n} unavailable={u}"


def num(value, spec, scale=1.0):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "n/a"
    return format(scale * value, spec)


def _load(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def summary(root):
    root = Path(root)
    lines = []
    status_dir = root / "status"
    for st in sorted(status_dir.iterdir()) if status_dir.is_dir() else ():
        try:
            text = st.read_text().strip()
        except OSError:
            text = "unreadable"
        lines.append(f"{st.name:40s} {text}")
    for arm in ARMS:
        for seed in SEEDS:
            line = f"{arm:30s} {seed}"
            g = root / "profiles" / f"{arm}-{seed}-gpu.json"
            gp = _load(g) if g.exists() else None
            if isinstance(gp, dict):
                steady = gp.get("steady_step_seconds")
                step = steady.get("median") if isinstance(steady, dict) else None
                line += f"  peak {num(gp.get('peak_gib'), '.2f')} GiB  step {num(step, '.3f')} s"
            elif g.exists():
                line += "  gpu profile unreadable"
            r = root / arm / str(seed) / "task_result.json"
            v = _load(r) if r.exists() else None
            if isinstance(v, dict):
                val = v.get("validation") if isinstance(v.get("validation"), dict) else {}
                line += (f"  selected epoch {v.get('selected_epoch', 'n/a')}"
                         f"  val acc {num(val.get('accuracy'), '.4f', 100)}%"
                         f"  CE {num(val.get('cross_entropy'), '.4f')}")
            elif r.exists():
                line += "  task_result unreadable"
            lines.append(line)
    return "\n".join(lines)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2 or argv[0] not in ("p-audit-status", "summary"):
        print(__doc__, file=sys.stderr)
        return 2
    if argv[0] == "p-audit-status":
        status = p_audit_status(argv[1])
        print(status)
        return 1 if status.startswith("FAIL") else 0
    try:
        print(summary(argv[1]))
    except Exception as exc:   # reporting never decides the job outcome
        print(f"summary unavailable: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
