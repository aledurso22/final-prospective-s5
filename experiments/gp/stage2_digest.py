"""Print a compact digest of a Stage 2 diagnostic output directory.

Read-only: parses the JSON the diagnostic already wrote. No model, no data, no
GPU, no numerics beyond formatting. Safe to run anywhere the files are.

    python -m experiments.gp.stage2_digest <diagnostic_output_dir>
"""

import json
import os
import sys

ARMS = ("native_s5", "alpha_p_s5", "gain_clip_s5", "gp_fixed_m0",
        "gp_fixed_mass")


def load(d, name):
    p = os.path.join(d, name)
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def main():
    d = sys.argv[1]
    print(f"=== diagnostic digest: {d}")

    s = load(d, "summary.json")
    if s:
        print(f"status={s['status']} wall={s['wall_s']:.1f}s "
              f"budget={s['budget_s']:.0f}s backend={s['backend']}")
        print(f"source_hashes_unchanged={s['source_hashes_unchanged']} "
              f"runtime_error={s.get('runtime_error')} "
              f"interrupted={s.get('interrupted')}")
        print(f"failed_required={[c['check'] for c in s['failed_required_checks']]}")
        print(f"unexecuted={[(c['check'], c.get('reason')) for c in s['unexecuted_checks']]}")

    rc = load(d, "restore_checks.json") or []
    print("\n--- RESTORE (saved vs restored, exact counts) ---")
    for r in rc:
        print(f"  {r['arm']:14s} n={r['n']} saved={r['saved_correct']} "
              f"restored={r['restored_correct']} match={r['counts_match']} "
              f"ce_diff={r['ce_abs_diff']:.2e} acc={r['restored_accuracy']*100:.2f}% "
              f"ckpt_epoch={r['checkpoint_epoch']} dtypes={r['dtypes']['params_dtypes']}")

    ac = load(d, "adapter_checks.json") or []
    print("\n--- ADAPTER vs EXECUTED CORE (all layers, all inputs) ---")
    for arm in ARMS:
        rows = [a for a in ac if a["arm"] == arm]
        if rows:
            print(f"  {arm:14s} layers={len(rows)} worst_rel="
                  f"{max(a['worst_rel'] for a in rows):.2e} "
                  f"inputs={rows[0]['n_inputs_checked']} "
                  f"pass={all(a['passed'] for a in rows)}")

    fd = load(d, "fd_checks.json") or []
    print("\n--- JVP vs FINITE DIFFERENCE ---")
    for f in fd:
        ps = f.get("per_step", {})
        extra = " ".join(f"{k}:{v['rel_error']:.2e}" for k, v in ps.items())
        print(f"  {f['arm']:14s} rel={f['rel_error']:.2e} pass={f['passed']}  {extra}")

    cores = load(d, "cores.json") or []
    print("\n--- CORE RESPONSE (per arm, per layer) ---")
    print(f"  {'arm':14s} {'L':>1} {'rho':>9} {'clip':>5} {'tau_med':>9} "
          f"{'|K0|':>8} {'|D|':>8} {'|K0-D|':>8} {'cf_rel':>7}")
    for c in cores:
        p = c["poles"]
        cf = c.get("counterfactual_one_tap", {}).get("rel_change_vs_executed")
        print(f"  {c['arm']:14s} {c['layer']:1d} "
              f"{c['spectral_radius_descriptive_only']:9.6f} "
              f"{p['n_raw_poles_clipped']:5d} {p['decay_frames_median']:9.1f} "
              f"{c['K0_norm']:8.3f} {c['D_norm']:8.3f} {c['K0_minus_D_norm']:8.3f} "
              f"{(f'{cf:.4f}' if cf is not None else '-'):>7}")

    print("\n--- IMPULSE ENERGY BY LAG BAND (layer 0, absolute | fraction) ---")
    for c in cores:
        if c["layer"] != 0:
            continue
        b = c["impulse_bands"]
        cells = " ".join(f"{r['lo']}-{r['hi']}:{r['energy']:.3e}/{r['fraction']:.3f}"
                         for r in b["bands"])
        print(f"  {c['arm']:14s} total={b['total_energy']:.4e}  {cells}")
        print(f"                 window={c['window_lags']} remainder={c['window_remainder']}")

    print("\n--- CONTINUOUS POLES (layer 0) ---")
    for c in cores:
        if c["layer"] != 0:
            continue
        p = c["poles"]
        re = p["continuous_pole_re"]; im = p["continuous_pole_im"]
        print(f"  {c['arm']:14s} Re[{min(re):+.5f},{max(re):+.5f}] "
              f"|Im|max={max(abs(v) for v in im):.4f} "
              f"aliased={p['n_continuous_modes_aliased_by_discrete_angle']} "
              f"delta=[{p['delta_min']:.4f},{p['delta_max']:.4f}] "
              f"raw_vs_clip={p['raw_minus_clipped_max']:.3e} "
              f"trained_vs_init={p['trained_minus_initializer_max']:.3e}")

    ic = load(d, "init_cores.json") or []
    if ic:
        print("\n--- RECONSTRUCTED INITIALIZATION (layer 0) ---")
        for c in ic:
            if c["layer"] != 0:
                continue
            b = c["impulse_bands"]
            print(f"  {c['arm']:14s} total={b['total_energy']:.4e} "
                  f"tau_med={c['poles']['decay_frames_median']:.1f} "
                  f"clip={c['poles']['n_raw_poles_clipped']}")

    fr = load(d, "frequency") or None
    g = load(d, "gradients.json") or []
    print("\n--- PARAMETER GRADIENTS (inference mode) ---")
    for r in g:
        if r.get("training_mode"):
            continue
        pl = r["per_layer"]
        per = " ".join(f"{k.replace('layers_','L')}:{v['grad_norm']:.3e}"
                       for k, v in sorted(pl.items()))
        print(f"  {r['arm']:14s} loss={r['loss']:.4f} total={r['total_grad_norm']:.4e} {per}")

    ba = load(d, "block_activations.json") or []
    print("\n--- ACTIVATION GRADIENTS, per-example norms (mean over batch) ---")
    for r in ba:
        print(f"  {r['arm']}: loss={r['loss']:.4f} [{r['offsets']}]")
        for s_ in r["sites"]:
            lab = f"{s_['site']}" + (f" L{s_['layer']}" if s_["layer"] is not None else "")
            print(f"      {lab:42s} pe_mean={s_['per_example_grad_norm_mean']:.4e} "
                  f"pe_rms={s_['per_example_grad_rms']:.4e} "
                  f"shared_equiv={s_['shared_offset_equivalent_norm']:.4e} "
                  f"skip={s_['includes_identity_skip']}")
        for a_ in r["activations"]:
            print(f"      L{a_['layer']} block_out_rms={a_['block_output_rms']:.4f} "
                  f"core_out_rms={a_['recurrent_core_output_rms']:.4f}")

    se = load(d, "sensitivity.json") or []
    print("\n--- PRE-POOLING SENSITIVITY BY LAG BAND (mean over 4 directions) ---")
    for r in se:
        for a_ in r["anchors"]:
            ds = a_["directions"]
            nb = len(ds[0]["bands"])
            avg = [sum(d["bands"][i]["fraction"] for d in ds) / len(ds)
                   for i in range(nb)]
            lab = " ".join(f"{ds[0]['bands'][i]['lo']}-{ds[0]['bands'][i]['hi']}:"
                           f"{avg[i]:.4f}" for i in range(nb)
                           if ds[0]['bands'][i]['in_history'])
            fut = max(d["future_sensitivity_max"] for d in ds)
            tot = sum(d["total_energy"] for d in ds) / len(ds)
            print(f"  {r['arm']:14s} anchor={a_['anchor']:3d} total={tot:.4e} "
                  f"future={fut:.1e}  {lab}")


if __name__ == "__main__":
    main()
