"""Digest for a Nesterov/QHM ladder run. Written BEFORE the run.

Read-only: saved JSON only; no JAX, no model, no recomputation.

    python -m experiments.prospective_momentum.nesterov_ladder_summary <run_dir>
"""

import json
import os
import sys

#: user-facing display order (internal identifiers)
LADDER_ORDER = ("native_full", "tss_processing", "operator_full",
                "generalized_processing", "literal_nesterov", "qhm")
PRIMARY_METRICS = ("revision", "retention", "recall", "immediate_revised",
                   "later_revised")
SHOW = ("revision", "retention", "recall", "immediate_revised",
        "later_revised", "later", "untouched_keys", "revised_idle_gap",
        "revised_intervening_writes", "untouched_idle_gap",
        "untouched_intervening_writes")


def fmt(v, nd=4):
    return "   n/a" if v is None else f"{v:+.{nd}f}"


def main(run_dir):
    st = json.load(open(os.path.join(run_dir, "status.json")))
    res_path = os.path.join(run_dir, "results.json")
    res = json.load(open(res_path)) if os.path.isfile(res_path) else None
    print(f"run {st.get('run_id')} complete={st.get('complete')} "
          f"failed={st.get('failed')} incomplete={st.get('incomplete')} "
          f"wall={st.get('wall_s')}")
    print(f"study: {st.get('study')}")
    names = st.get("scientific_names") or {}
    for i, arm in enumerate(st.get("display_order") or LADDER_ORDER, 1):
        print(f"  {i}. {names.get(arm, arm)}")
    print(f"scope: {st.get('generalized_scope')}")
    print(f"extra parameters: {st.get('extra_parameters')}")
    print(f"executed carry (reals): {st.get('carry_executed')}")
    print(f"extra per-token work: {st.get('extra_work')}")
    print(f"declared departures: {st.get('declared_departures')}")
    pf = st.get("preflight") or {}
    print(f"\npreflight projected {pf.get('projected_remaining_s')} s, "
          f"retraced {pf.get('retraced_any')}, failures {pf.get('failures')}")
    for r in pf.get("rows", []):
        print(f"  {names.get(r['arm'], r['arm']):<64} step "
              f"{r['step_s'] * 1e3:7.2f} ms  checkpoint "
              f"{r['checkpoint_s']:5.2f} s  eval {r['evaluation_s']:5.2f} s")
    for seed, s in (st.get("start_points") or {}).items():
        t = s.get("nesterov_start_table") or {}
        print(f"start {seed}: nesterov tree is native "
              f"{s.get('nesterov_tree_is_native_tree')}; recovery failures "
              f"{s.get('recovery_failures')}; nesterov table "
              f"{t.get('classification')} closed form "
              f"{t.get('closed_form_condition')}")
    sel = st.get("selection") or {}
    print(f"\nselection rule: {sel.get('rule')}")
    print(f"unavailable: {sel.get('unavailable')}")
    for row in sel.get("table", []):
        c = row.get("chosen")
        print(f"  {names.get(row['arm'], row['arm']):<64} " + (
            "UNAVAILABLE" if c is None else
            f"lr {c['lr']} update {c['update']:>3} revision {c['primary']:.4f}"
            f" immediate {c['immediate_revision']:.4f} later {c['later']:.4f}")
            + f"  excluded(unstable table) "
            f"{row.get('excluded_frozen_token_unstable')}")
    if res is None:
        print("\nno results.json (run not complete)")
        return
    print("\n=== held-out metrics per arm and seed ===")
    print("  " + "arm/seed".ljust(34)
          + "".join(m[:10].rjust(11) for m in SHOW))
    rnames = res.get("scientific_names") or names
    for key in sorted(res["heldout"], key=lambda k: (
            LADDER_ORDER.index(k.split("/")[0])
            if k.split("/")[0] in LADDER_ORDER else 99, k)):
        v = res["heldout"][key]
        arm, seed = key.split("/")
        print("  " + f"{rnames.get(arm, arm)} / seed {seed}")
        print("  " + " " * 34 + "".join(f"{v[m]:11.4f}" for m in SHOW))
    ap = res.get("nesterov_applicability") or {}
    print(f"\n=== Literal Nesterov Momentum DeltaNet applicability "
          f"({ap.get('condition')}) ===")
    print(f"  checkpoints evaluated {ap.get('checkpoints_evaluated')}, "
          f"unstable {ap.get('checkpoints_unstable')} "
          f"(fraction {ap.get('unstable_fraction')}); unavailable "
          f"{ap.get('unavailable')}")
    for u in ap.get("unstable", []):
        print(f"    UNSTABLE {u['stage']}/{u['config']}/seed{u['seed']}/"
              f"u{u['update']}: {u['classification']} min Jury "
              f"{u['min_jury_expression']} closed form {u['closed_form']}")
    print(f"  eligibility basis: {ap.get('eligibility_basis')}")
    for seed, e in (ap.get("heldout_endpoints") or {}).items():
        print(f"  held-out endpoint seed {seed}: table "
              f"{(e.get('endpoint_table') or {}).get('classification')}; "
              f"FAILURES RETAINED IN THE DENOMINATOR "
              f"{e.get('failures_retained_in_denominator')}; realized-gate "
              f"record {e.get('episodes')}")
    pa = res["paired_analysis"]
    print("\nplanned primary contrasts:")
    for name in pa["planned_primary_contrasts"]:
        first, second = name.split("_vs_", 1)
        print(f"  {rnames.get(first, first)} vs {rnames.get(second, second)}")
    print(f"\n=== SECONDARY DIAGNOSTIC: stable-subset exclusions ===")
    for seed, e in pa["stable_subset_exclusions"].items():
        print(f"  seed {seed}: excluded {e['excluded_episodes']} of "
              f"{e['episodes']} episodes ({e['excluded_fraction']:.4f}), "
              f"{e['excluded_blocks']} of {e['blocks']} blocks; by cell "
              f"{e['by_family_and_condition']}")
    for tag, key in (("full", "full"),
                     ("stable_subset", "mechanism_diagnostic_stable_subset")):
        part = pa[key]
        print(f"\n=== paired analysis on {tag.upper()} "
              f"({'PRIMARY' if tag == pa['primary'] else 'SECONDARY MECHANISM DIAGNOSTIC, not used for the recommendation'}) ===")
        print(f"  not computable: {part['not_computable']}")
        for name, c in part["comparisons"].items():
            first, second = name.split("_vs_", 1)
            print(f"\n  {rnames.get(first, first)} vs "
                  f"{rnames.get(second, second)}  [{c.get('role')}]")
            for m in SHOW:
                x = c[m]
                print(f"    {m:<30} D {fmt(x['D'])}  CI95 [{fmt(x['ci95'][0])}"
                      f", {fmt(x['ci95'][1])}]  SEw {x['se_within']:.4f}  "
                      f"SEb {fmt(x['se_between_seeds'])}  per-seed "
                      f"{[round(d, 4) for d in x['per_seed'].values()]} "
                      f"signs {''.join(x['per_seed_sign'].values())} "
                      f"-> {x['label']}")
        print(f"\n  recommendation on {tag}: "
              f"{part.get('recommendation') or part.get('diagnostic_recommendation')}"
              + ("" if tag == pa["primary"] else " (diagnostic only)"))
        print(f"  immediate claim on {tag}: {part['immediate_claim']}")
    print("\n=== PLANNED PRIMARY CONTRASTS (complete held-out set): "
          "Generalized prospective dynamics (M,\u03b3,T) versus ... ===")
    print(f"  scope: {pa.get('generalized_scope')}")
    for name in pa["planned_primary_contrasts"]:
        c = pa["full"]["comparisons"].get(name)
        other = name.split("_vs_", 1)[1]
        print(f"\n  versus {rnames.get(other, other)}")
        if c is None:
            print("    not computable")
            continue
        for m in PRIMARY_METRICS:
            x = c[m]
            print(f"    {m:<18} D {fmt(x['D'])}  SE {x['se_within']:.4f}  CI95 "
                  f"[{fmt(x['ci95'][0])}, {fmt(x['ci95'][1])}]  seed signs "
                  f"{''.join(x['per_seed_sign'].values())} -> {x['label']}")
    print(f"\nRECOMMENDATION (primary): {pa['recommendation']}")
    print(f"basis: {pa['recommendation_basis']}")


if __name__ == "__main__":
    main(sys.argv[1])
