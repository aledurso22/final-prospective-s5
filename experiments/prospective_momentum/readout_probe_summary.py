"""Digest for a Stage B readout-probe run. Written BEFORE the run.

Read-only: saved JSON only; no JAX, no model, no recomputation.

    python -m experiments.prospective_momentum.readout_probe_summary <run_dir>
"""

import json
import os
import sys

from .summary import f


def fmt(v, nd=4):
    return "  n/a" if v is None else f"{v:.{nd}f}"


def table(metrics, arms, categories, conditions, offsets):
    """One block per category and condition: arms as rows, offsets as
    columns, best per column marked."""
    for cat in categories:
        for cond in conditions:
            keys = [f"{cat}/{cond}/k{k}" for k in offsets]
            best = {}
            for key in keys:
                vals = [(n, metrics[n][key]["mean"]) for n in arms
                        if metrics[n][key]["mean"] is not None]
                best[key] = min(vals, key=lambda x: x[1])[0] if vals else None
            print(f"\n  [{cat} | {cond}]  (1.0 = no better than native)")
            print("    " + "arm".ljust(38)
                  + "".join(f"k={k}".rjust(10) for k in offsets))
            for n in arms:
                row = "".join(
                    (("*" if best[key] == n else " ")
                     + fmt(metrics[n][key]["mean"]).rjust(9))
                    for key in keys)
                print("    " + n.ljust(38) + row)


def main(run_dir):
    st = json.load(open(os.path.join(run_dir, "status.json")))
    print(f"run {st.get('run_id')} [{st.get('study')}] complete="
          f"{st.get('complete')} failed={st.get('failed')} incomplete="
          f"{st.get('incomplete')} wall={st.get('wall_s')} source_unchanged="
          f"{st.get('source_unchanged')}")
    print(f"computation_status={st.get('computation_status')} study_status="
          f"{st.get('study_status')} integrity_verified="
          f"{st.get('integrity_verified')}")
    print(f"training: {st.get('training')}")
    print(f"measurement: {st.get('measurement')}")
    print(f"checkpoints {st.get('checkpoint_run')} seeds {st.get('seeds')}")
    print(f"grids {st.get('grids')}")
    print(f"arms {st.get('n_arms')} stream {st.get('stream')} episodes/family "
          f"{st.get('episodes_per_family')} offsets {st.get('offsets')}")
    tc = st.get("task_check") or {}
    print(f"task: queries {tc.get('queries_per_sequence')} balanced "
          f"{tc.get('block_cells_balanced')} oracle {tc.get('oracle_exact')} "
          f"digest {st.get('task_digest')}")
    ref = st.get("reference") or {}
    print("\n--- frozen checkpoints (read-only) ---")
    for s, path in (ref.get("files") or {}).items():
        print(f"  seed {s}: {path}")
    print(f"  reproduction ok: "
          f"{all(v['ok'] for r in (ref.get('reproduction') or {}).values() for v in r.values())}")

    arms = st.get("arms", [])
    pooled = (st.get("pooled") or {}).get("metrics")
    offsets = st.get("offsets", [])
    if pooled:
        print("\n=== pooled over seeds ===")
        table(pooled, arms, ("overall", "revised_direction",
                             "untouched_direction"),
              ("both", "idle_gap", "intervening_writes"), offsets)
        print("\n--- best arm per category, condition and offset ---")
        for key, v in ((st.get("pooled") or {}).get("argmins") or {}).items():
            print(f"  {key:<34} {v and v['arm']:<38} {fmt(v and v['mean'])}")
        sr = (st.get("pooled") or {}).get("stopping_rule") or {}
        print("\n--- PREDECLARED STOPPING RULE ---")
        print(f"  margin {sr.get('margin')} offsets required "
              f"{sr.get('offsets_required')} won {sr.get('offsets_won')}")
        for key, row in (sr.get("per_offset") or {}).items():
            print(f"  {key}: interior {row.get('generalized_interior')} "
                  f"tss {row.get('literal_tss')} difference "
                  f"{fmt(row.get('difference'))} wins {row.get('wins')}")
        print(f"  interior_beats_literal_tss="
              f"{sr.get('interior_beats_literal_tss')}")
        print(f"  {sr.get('verdict')}")
        print(f"  {sr.get('note')}")

    for seed, r in (st.get("per_seed") or {}).items():
        print(f"\n=== seed {seed} (wall {f(r.get('wall_s'), 1)}s, native-point "
              f"agreement {r.get('native_point_logit_agreement')}) ===")
        sr = r.get("stopping_rule") or {}
        print(f"  stopping rule: won {sr.get('offsets_won')} of "
              f"{sr.get('offsets_required')} required -> "
              f"{sr.get('interior_beats_literal_tss')}")
        print(f"  dW components: {r.get('component_split')}")
        for key, v in (r.get("argmins") or {}).items():
            print(f"    argmin {key:<32} {v and v['arm']:<38} "
                  f"{fmt(v and v['mean'])}")
        sec = r.get("secondary") or {}
        for name in ("native_identity",):
            if name in sec and sec[name]:
                print(f"    secondary {name}: revised "
                      f"{[round(x, 4) for x in sec[name]['revised_label_probability'][:6]]}"
                      f" untouched "
                      f"{[round(x, 4) for x in sec[name]['untouched_label_probability'][:6]]}")


if __name__ == "__main__":
    main(sys.argv[1])
