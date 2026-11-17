"""Trained native vs trained generalized S5, each against its matched init.

Lightweight and checkpoint-only: no training, no data, no random search.

INPUT. Task directories as the S5 launchers write them,

    <run_root>/<arm>/<seed>/checkpoint_epoch_NN.msgpack
    <run_root>/<arm>/<seed>/task_result.json        (seed, arm, selected_epoch)
    <run_root>/run_metadata.txt                     (commit, scan, ...)

passed as --native TASK_DIR... and --generalized TASK_DIR... . The two lists
are paired BY SEED, and every seed must appear in both. --epoch is either
"selected" (each task's own selected_epoch, from task_result.json) or an
integer used for every task. Anything unmatched or unsupported fails loudly:
a missing file, a seed or arm that disagrees with task_result.json, a
parameter tree whose shapes differ from runner.init_state(arm, seed), or a
native checkpoint passed as generalized (or the reverse).

WHAT IS COMPARED, per layer and mode, for TRAINED and MATCHED INIT:
  * Lambda (clipped as in S5SSM) and Delta = exp(log_step) JOINTLY: the
    product Lambda*Delta, and the effective discrete poles they produce;
  * native: the pole lambda_bar -> decay time -1/log|z| and rotation arg z;
  * generalized (the arm that was actually trained, legacy k=T, old
    corrector): both companion roots, their decay times and rotations,
    T, gamma, rho, M, T/gamma = eta/(1-mu), and the partial-fraction residues
    of each component (per unit B_bar);
  * learned B/C coupling: per mode ||B_bar_p|| (over inputs) and ||C_p||
    (over outputs, forward C1 and backward C2), and a component gain proxy
    ||C_p|| ||B_bar_p|| |R_i| / sqrt(1 - |z_i|^2) -- the l2 impulse norm of
    component i alone. Slow/fast interference terms are omitted, so this is
    a COMPONENT proxy, not an exact share of the mode's output energy.

TWO SEPARATE QUESTIONS, never mixed in the output:
  * "actual": trained native vs trained generalized (both really trained);
  * "counterfactual": variants C and D (and A, recomputed as a check)
    evaluated AT THE GENERALIZED ARM'S LEARNED PARAMETERS. Nothing was
    trained with those coefficients.

Compensation is NOT inferred from log_step alone: the report states the
effective slow-pole decay times and rotations of both trained arms.

STABLE MODES ONLY for response statistics. Each variant reports its count of
stable (Schur-Cohn) and unstable modes. DC delay, residue ratios, the energy
proxy, the exact-total energy share and the component gains are computed over
STABLE modes only (keys end in `_stable_modes_only`): for an unstable mode
those infinite-horizon quantities do not exist.

INITIALIZATION AGREEMENT. `reconstructed_ssm_init_identical_across_arms`
compares the SSM-layer leaves (Lambda, log_step, B, C1, C2) of the two arms'
initializations as RECONSTRUCTED today by `runner.init_state(arm, seed)`
in float32. It does NOT verify that the historical runs started from that
reconstruction, and it does not compare the rest of the backbone (encoder,
norms, decoder); no initial checkpoint was stored to check against.

OUTPUT IS NEVER OVERWRITTEN: if <out>/checkpoint_compare.json exists the
tool refuses; choose a new --out.

    python -m experiments.s5_predictor_corrector.checkpoint_compare \
        --native  RUN_N/native_matched_s5/301 RUN_N/native_matched_s5/302 ... \
        --generalized RUN_G/generalized_prospective_s5/301 ... \
        --epoch selected --out results/s5_predictor_corrector/trained
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as onp

NATIVE, GENERALIZED = "native_matched_s5", "generalized_prospective_s5"
GEN_KEYS = ("generalized_T_raw", "generalized_rho_raw", "generalized_gamma_raw")
LAYER_KEYS = ("Lambda_re", "Lambda_im", "log_step", "B", "C1", "C2")
PCT = (0, 5, 25, 50, 75, 95, 100)


class CheckpointError(ValueError):
    """Unmatched, inconsistent or unsupported checkpoint input."""


def q(values):
    v = onp.asarray(values, float).ravel()
    v = v[onp.isfinite(v)]
    if v.size == 0:
        return None
    return {f"p{p}": float(onp.percentile(v, p)) for p in PCT}


# ------------------------------------------------------------- inputs ---

def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_metadata(task_dir):
    path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(task_dir))), "run_metadata.txt")
    if not os.path.exists(path):
        return None
    out = {}
    with open(path) as fh:
        for line in fh:
            if "=" in line:
                k, v = line.rstrip("\n").split("=", 1)
                out[k] = v
    return out


def resolve_task(task_dir, arm, epoch):
    """(checkpoint path, provenance) for one task directory, or raise."""
    if not os.path.isdir(task_dir):
        raise CheckpointError(f"{arm}: not a directory: {task_dir}")
    result_path = os.path.join(task_dir, "task_result.json")
    result = None
    if os.path.exists(result_path):
        with open(result_path) as fh:
            result = json.load(fh)
    seed_dir = os.path.basename(os.path.normpath(task_dir))
    arm_dir = os.path.basename(os.path.dirname(os.path.abspath(task_dir)))
    if arm_dir != arm:
        raise CheckpointError(
            f"{task_dir}: parent directory is {arm_dir!r}, expected {arm!r}")
    if not seed_dir.isdigit():
        raise CheckpointError(f"{task_dir}: last component is not a seed")
    seed = int(seed_dir)
    if result is not None:
        if result.get("code_identifier") != arm:
            raise CheckpointError(
                f"{result_path}: code_identifier "
                f"{result.get('code_identifier')!r} != {arm!r}")
        if int(result.get("seed", -1)) != seed:
            raise CheckpointError(
                f"{result_path}: seed {result.get('seed')} != directory {seed}")
    if epoch == "selected":
        if result is None or "selected_epoch" not in result:
            raise CheckpointError(
                f"{task_dir}: --epoch selected needs task_result.json with "
                "selected_epoch; pass an explicit --epoch instead")
        ep = int(result["selected_epoch"])
    else:
        ep = int(epoch)
    ckpt = os.path.join(task_dir, f"checkpoint_epoch_{ep:02d}.msgpack")
    if not os.path.exists(ckpt):
        raise CheckpointError(f"{arm} seed {seed}: missing {ckpt}")
    prov = dict(arm=arm, seed=seed, epoch=ep, epoch_rule=str(epoch),
                task_dir=os.path.abspath(task_dir), checkpoint=os.path.abspath(ckpt),
                checkpoint_sha256=_sha256(ckpt),
                checkpoint_bytes=os.path.getsize(ckpt),
                selected_epoch=None if result is None else result.get("selected_epoch"),
                run_metadata=_run_metadata(task_dir))
    return ckpt, prov


def read_params(path):
    from flax import serialization
    with open(path, "rb") as fh:
        restored = serialization.msgpack_restore(fh.read())
    if not isinstance(restored, dict) or "params" not in restored:
        raise CheckpointError(f"{path}: not a TrainState checkpoint (no 'params')")
    return restored["params"]


def layer_groups(params, arm, where):
    """Per SSM layer: the raw leaves, validated for the arm. Accepts a nested
    parameter tree or an already-flat {tuple path: array} dict."""
    from flax.traverse_util import flatten_dict
    flat = (params if params and all(isinstance(k, tuple) for k in params)
            else flatten_dict(params))
    prefixes = sorted({k[:-1] for k in flat if k[-1] == "Lambda_re"})
    if not prefixes:
        raise CheckpointError(f"{where}: no S5 layers (no Lambda_re leaves)")
    groups = {}
    for p in prefixes:
        g = {k[-1]: onp.asarray(v) for k, v in flat.items() if k[:-1] == p}
        missing = [k for k in LAYER_KEYS if k not in g]
        if missing:
            raise CheckpointError(f"{where}: layer {'/'.join(p)} lacks {missing}")
        has_gen = all(k in g for k in GEN_KEYS)
        if arm == GENERALIZED and not has_gen:
            raise CheckpointError(
                f"{where}: passed as {GENERALIZED} but layer {'/'.join(p)} has "
                "no generalized_* parameters (a native checkpoint?)")
        if arm == NATIVE and any(k in g for k in GEN_KEYS):
            raise CheckpointError(
                f"{where}: passed as {NATIVE} but carries generalized_* parameters")
        groups["/".join(p)] = g
    return groups


def check_against_init(groups, init_groups, where):
    if sorted(groups) != sorted(init_groups):
        raise CheckpointError(f"{where}: layer set differs from init_state")
    for name, g in groups.items():
        for k, v in g.items():
            if k not in init_groups[name] or v.shape != init_groups[name][k].shape:
                raise CheckpointError(
                    f"{where}: {name}/{k} shape {v.shape} does not match the "
                    "matched init (unsupported architecture?)")


# ----------------------------------------------------------- analysis ---

def _complex(x):
    return x[..., 0].astype(onp.float64) + 1j * x[..., 1].astype(onp.float64)


def mode_quantities(g, clip_eigs=True):
    """Lambda, Delta, lambda_bar, B_bar row norms, C column norms."""
    re = onp.asarray(g["Lambda_re"], onp.float64)
    lam = (onp.minimum(re, -1e-4) if clip_eigs else re) + 1j * onp.asarray(
        g["Lambda_im"], onp.float64)
    delta = onp.exp(onp.asarray(g["log_step"][:, 0], onp.float64))
    lb = onp.exp(lam * delta)
    b_tilde = _complex(g["B"])                       # (P, H)
    b_bar_scale = (lb - 1.0) / lam                   # B_bar = scale * B_tilde
    b_norm = onp.abs(b_bar_scale) * onp.linalg.norm(b_tilde, axis=1)
    c1 = onp.linalg.norm(_complex(g["C1"]), axis=0)  # (P,) over outputs
    c2 = onp.linalg.norm(_complex(g["C2"]), axis=0)
    return dict(Lambda=lam, Delta=delta, LambdaDelta=lam * delta,
                lambda_bar=lb, b_bar_norm=b_norm, c1_norm=c1, c2_norm=c2,
                c_norm=onp.sqrt(c1 ** 2 + c2 ** 2))


def generalized_params(g):
    from experiments.s5_two_compartment.parameter_sample import leaves
    T, rho, gamma, M = leaves(g)
    return T, M, gamma, rho


def pole_stats(z):
    r = onp.abs(z)
    with onp.errstate(divide="ignore"):
        decay = onp.where(r < 1, -1.0 / onp.log(r), onp.inf)
    return r, decay, onp.angle(z)


def analyse_native(groups):
    out = []
    for name, g in groups.items():
        m = mode_quantities(g)
        r, decay, rot = pole_stats(m["lambda_bar"])
        gain = m["c_norm"] * m["b_bar_norm"] / onp.sqrt(onp.maximum(1 - r ** 2, 1e-300))
        out.append(dict(layer=name, **m, radius=r, decay=decay, rotation=rot,
                        component_gain=gain))
    return out


def analyse_generalized(groups):
    from s5 import predictor_corrector as PC
    from s5.discrete_recurrence import generalized_coefficients
    out = []
    for name, g in groups.items():
        m = mode_quantities(g)
        T, M, gamma, rho = generalized_params(g)
        lb = m["lambda_bar"]
        prod = generalized_coefficients(lb, onp.ones((lb.size, 1)), T, M, gamma)
        prod = (prod[0], prod[1], prod[2][:, 0], prod[3][:, 0])
        rec = {}
        for v, (scale, corr) in PC.VARIANTS.items():
            co = PC.scalar_coefficients(lb, T, M, gamma, scale, corr)
            z1, z2, R1, R2 = PC.residues(*co)
            r1, d1, w1 = pole_stats(z1)
            r2, d2, w2 = pole_stats(z2)
            stable = PC.schur_cohn(co[0], co[1])
            with onp.errstate(divide="ignore", invalid="ignore"):
                e1 = onp.abs(R1) ** 2 / (1 - r1 ** 2)
                e2 = onp.abs(R2) ** 2 / (1 - r2 ** 2)
                total = e1 + e2 + 2 * (R1 * onp.conj(R2) / (1 - z1 * onp.conj(z2))).real
            k = PC.target_scale(T, gamma, scale)
            mu, eta, zeta = PC.residual_coefficients(T, M, gamma, k)
            rec[v] = dict(stable=stable, slow=z1, fast=z2, R_slow=R1, R_fast=R2,
                          slow_radius=r1, slow_decay=d1, slow_rotation=w1,
                          fast_radius=r2, fast_decay=d2, fast_rotation=w2,
                          dc_delay=PC.group_delay_dc(*co),
                          eta_over_1_minus_mu=eta / (1 - mu), abs_P=onp.abs(1 - eta * (1 - lb)),
                          eta_absK=eta * onp.abs(1 - lb),
                          disc_ratio=onp.abs(co[0] ** 2 + 4 * co[1]) / onp.abs(co[0]) ** 2,
                          fast_component_energy_proxy=e2 / (e1 + e2),
                          fast_energy_over_exact_total=e2 / total,
                          slow_component_gain=m["c_norm"] * m["b_bar_norm"] * onp.sqrt(
                              onp.where(stable, e1, onp.nan)),
                          fast_component_gain=m["c_norm"] * m["b_bar_norm"] * onp.sqrt(
                              onp.where(stable, e2, onp.nan)))
            if v == "A":
                rec[v]["max_abs_diff_vs_production_coefficients"] = float(max(
                    onp.max(onp.abs(onp.asarray(a) - onp.asarray(b)))
                    for a, b in zip(co, prod)))
        out.append(dict(layer=name, **m, T=T, M=M, gamma=gamma, rho=rho,
                        T_over_gamma=T / gamma, variants=rec))
    return out


def native_summary(layers):
    cat = lambda k: onp.concatenate([l[k] for l in layers])
    return dict(
        n_modes=int(cat("radius").size),
        Lambda_re=q(cat("Lambda").real), Lambda_im=q(cat("Lambda").imag),
        Delta=q(cat("Delta")), log_Delta=q(onp.log(cat("Delta"))),
        LambdaDelta_re=q(cat("LambdaDelta").real),
        LambdaDelta_im=q(cat("LambdaDelta").imag),
        pole_radius=q(cat("radius")), decay_tokens=q(cat("decay")),
        rotation_rad_per_token=q(onp.abs(cat("rotation"))),
        b_bar_norm=q(cat("b_bar_norm")), c_norm=q(cat("c_norm")),
        component_gain=q(cat("component_gain")))


def generalized_summary(layers, variants):
    cat = lambda k: onp.concatenate([l[k] for l in layers])
    vcat = lambda v, k: onp.concatenate([l["variants"][v][k] for l in layers])
    base = dict(
        n_modes=int(cat("lambda_bar").size),
        Lambda_re=q(cat("Lambda").real), Lambda_im=q(cat("Lambda").imag),
        Delta=q(cat("Delta")), log_Delta=q(onp.log(cat("Delta"))),
        LambdaDelta_re=q(cat("LambdaDelta").real),
        LambdaDelta_im=q(cat("LambdaDelta").imag),
        backbone_lambda_bar_radius=q(onp.abs(cat("lambda_bar"))),
        T=q(cat("T")), gamma=q(cat("gamma")), rho=q(cat("rho")), M=q(cat("M")),
        T_over_gamma=q(cat("T_over_gamma")),
        b_bar_norm=q(cat("b_bar_norm")), c_norm=q(cat("c_norm")))
    for v in variants:
        sc = vcat(v, "stable").astype(bool)
        only = lambda k: vcat(v, k)[sc]
        radius = onp.maximum(vcat(v, "slow_radius"), vcat(v, "fast_radius"))
        rows = dict(
            # VALID = Schur-Cohn stable. Every *_stable_modes_only statistic
            # below is over these modes only; unstable modes have no
            # infinite-horizon energy, gain or DC delay.
            n_modes=int(sc.size), stable=int(sc.sum()), unstable=int((~sc).sum()),
            unstable_radius=q(radius[~sc]) if (~sc).any() else None,
            max_radius=float(radius.max()),
            slow_radius=q(vcat(v, "slow_radius")),
            fast_radius=q(vcat(v, "fast_radius")),
            slow_decay_tokens_stable_modes_only=q(only("slow_decay")),
            slow_rotation_rad_per_token=q(onp.abs(vcat(v, "slow_rotation"))),
            fast_decay_tokens_stable_modes_only=q(only("fast_decay")),
            residue_fast_over_slow_stable_modes_only=q(
                onp.abs(only("R_fast")) / onp.abs(only("R_slow"))),
            fast_component_energy_proxy_stable_modes_only=q(
                only("fast_component_energy_proxy")),
            fast_energy_over_exact_total_stable_modes_only=q(
                only("fast_energy_over_exact_total")),
            slow_component_gain_stable_modes_only=q(only("slow_component_gain")),
            fast_component_gain_stable_modes_only=q(only("fast_component_gain")),
            dc_delay_tokens_stable_modes_only=q(only("dc_delay")),
            eta_over_1_minus_mu=q(vcat(v, "eta_over_1_minus_mu")),
            abs_P=q(vcat(v, "abs_P")), eta_absK=q(vcat(v, "eta_absK")),
            min_discriminant_ratio=float(vcat(v, "disc_ratio").min()))
        if v == "A":
            rows["max_abs_diff_vs_production_coefficients"] = max(
                l["variants"]["A"]["max_abs_diff_vs_production_coefficients"]
                for l in layers)
        base[v] = rows
    return base


def paired(native_layers, gen_layers):
    """Per-mode ratios between the two arms at the same seed, layer, mode
    index. Mode identity is defined by the shared initialization only."""
    nd = onp.concatenate([l["decay"] for l in native_layers])
    nw = onp.abs(onp.concatenate([l["rotation"] for l in native_layers]))
    gd = onp.concatenate([l["variants"]["A"]["slow_decay"] for l in gen_layers])
    stable = onp.concatenate([l["variants"]["A"]["stable"] for l in gen_layers]).astype(bool)
    gd = onp.where(stable, gd, onp.nan)          # unstable: no decay time
    gw = onp.abs(onp.concatenate([l["variants"]["A"]["slow_rotation"] for l in gen_layers]))
    ndl = onp.concatenate([l["Delta"] for l in native_layers])
    gdl = onp.concatenate([l["Delta"] for l in gen_layers])
    with onp.errstate(divide="ignore", invalid="ignore"):
        return dict(slow_decay_generalized_over_native=q(gd / nd),
                    rotation_generalized_over_native=q(gw / nw),
                    Delta_generalized_over_native=q(gdl / ndl))


def drift(trained, init, key):
    a = onp.concatenate([l[key] for l in trained])
    b = onp.concatenate([l[key] for l in init])
    with onp.errstate(divide="ignore", invalid="ignore"):
        return q(a / b)


def compare_seed(native_ckpt, gen_ckpt, seed, init_flat):
    """init_flat(arm, seed) -> the float32 production init (x64-off draw)."""
    nat_init = layer_groups(init_flat(NATIVE, seed), NATIVE, f"init {NATIVE}/{seed}")
    gen_init = layer_groups(init_flat(GENERALIZED, seed), GENERALIZED,
                            f"init {GENERALIZED}/{seed}")
    nat = layer_groups(read_params(native_ckpt), NATIVE, native_ckpt)
    gen = layer_groups(read_params(gen_ckpt), GENERALIZED, gen_ckpt)
    check_against_init(nat, nat_init, native_ckpt)
    check_against_init(gen, gen_init, gen_ckpt)
    shared_init = all(
        onp.array_equal(nat_init[n][k], gen_init[n][k])
        for n in nat_init for k in LAYER_KEYS)
    an = {"trained": analyse_native(nat), "init": analyse_native(nat_init)}
    ag = {"trained": analyse_generalized(gen), "init": analyse_generalized(gen_init)}
    out = dict(seed=seed,
               reconstructed_ssm_init_identical_across_arms=bool(shared_init),
               init_agreement_scope=(
                   "SSM-layer leaves (Lambda, log_step, B, C1, C2) of today's "
                   "float32 runner.init_state reconstruction for both arms; NOT "
                   "a check of the historical runs' initial state, and not of "
                   "the encoder/norm/decoder leaves"))
    out["actual"] = dict(
        native=dict(trained=native_summary(an["trained"]),
                    matched_init=native_summary(an["init"]),
                    trained_over_init=dict(
                        Delta=drift(an["trained"], an["init"], "Delta"),
                        decay=drift(an["trained"], an["init"], "decay"))),
        generalized=dict(trained=generalized_summary(ag["trained"], ["A"]),
                         matched_init=generalized_summary(ag["init"], ["A"]),
                         trained_over_init=dict(
                             Delta=drift(ag["trained"], ag["init"], "Delta"),
                             T_over_gamma=drift(ag["trained"], ag["init"], "T_over_gamma"))),
        paired_trained=paired(an["trained"], ag["trained"]),
        paired_init=paired(an["init"], ag["init"]))
    full = generalized_summary(ag["trained"], ["A", "B", "C", "D"])
    out["counterfactual_at_generalized_learned_params"] = dict(
        note="B, C, D were NOT trained; A is the trained arm recomputed "
             "through the research module (max coefficient difference vs "
             "production reported)",
        **{v: full[v] for v in ("A", "B", "C", "D")})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--native", nargs="+", required=True, metavar="TASK_DIR")
    ap.add_argument("--generalized", nargs="+", required=True, metavar="TASK_DIR")
    ap.add_argument("--epoch", default="selected",
                    help='"selected" (per task, from task_result.json) or an integer')
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.epoch != "selected" and not args.epoch.isdigit():
        raise CheckpointError(f"--epoch must be 'selected' or an integer, got {args.epoch!r}")

    nat = {}
    for d in args.native:
        ck, pv = resolve_task(d, NATIVE, args.epoch)
        if pv["seed"] in nat:
            raise CheckpointError(f"native seed {pv['seed']} given twice")
        nat[pv["seed"]] = (ck, pv)
    gen = {}
    for d in args.generalized:
        ck, pv = resolve_task(d, GENERALIZED, args.epoch)
        if pv["seed"] in gen:
            raise CheckpointError(f"generalized seed {pv['seed']} given twice")
        gen[pv["seed"]] = (ck, pv)
    if sorted(nat) != sorted(gen):
        raise CheckpointError(
            f"unmatched seeds: native {sorted(nat)} vs generalized {sorted(gen)}")
    if os.path.exists(os.path.join(args.out, "checkpoint_compare.json")):
        raise CheckpointError(
            f"{os.path.join(args.out, 'checkpoint_compare.json')} exists; "
            "refusing to overwrite evidence. Choose a new --out.")

    # NO x64 here: the matched init must be the float32 production draw
    # (production_init draws it in an x64-off subprocess regardless); the
    # analysis itself is numpy float64.
    from experiments.s5_three_arm_full import runner as R
    from experiments.s5_predictor_corrector.production_init import (
        production_init_flat)
    from s5 import checkpointing
    seeds = sorted(nat)
    for s in seeds:
        if s not in R.SEEDS:
            raise CheckpointError(f"seed {s} is not a protocol seed {R.SEEDS}")
    report = dict(
        tool="experiments.s5_predictor_corrector.checkpoint_compare",
        analysis_provenance=checkpointing.provenance(),
        command=" ".join(sys.argv), epoch_rule=args.epoch,
        inputs=dict(native=[nat[s][1] for s in seeds],
                    generalized=[gen[s][1] for s in seeds]),
        seeds=[compare_seed(nat[s][0], gen[s][0], s, production_init_flat)
               for s in seeds])
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "checkpoint_compare.json")
    if os.path.exists(path):
        raise CheckpointError(f"{path} exists; refusing to overwrite evidence. "
                              "Choose a new --out.")
    with open(path, "w") as fh:
        json.dump(report, fh, indent=1, default=float)
    _print_digest(report)
    print("wrote", path)
    return report


def _print_digest(report):
    m = lambda d: "n/a" if d is None else f"{d['p50']:.4g} [{d['p5']:.3g}, {d['p95']:.3g}]"
    print("Diagnostics at the epoch analysed below. This is NOT a performance "
          "comparison, and it is not the selected-validation epoch unless "
          "they coincide:")
    for arm in ("native", "generalized"):
        for p in report["inputs"][arm]:
            md = p["run_metadata"] or {}
            print(f"  {arm:11s} seed {p['seed']} epoch {p['epoch']} "
                  f"(selected-validation epoch: {p['selected_epoch']}) "
                  f"commit {md.get('authoritative_commit')} "
                  f"scan {md.get('scan_implementation')}  {p['checkpoint']}")
    for s in report["seeds"]:
        a = s["actual"]
        print(f"== seed {s['seed']}  (reconstructed SSM init identical across "
              f"arms: {s['reconstructed_ssm_init_identical_across_arms']})")
        for arm in ("native", "generalized"):
            for when in ("matched_init", "trained"):
                d = a[arm][when]
                slow = d.get("decay_tokens") or d["A"]["slow_decay_tokens_stable_modes_only"]
                rot = d.get("rotation_rad_per_token") or d["A"]["slow_rotation_rad_per_token"]
                print(f"  ACTUAL {arm:11s} {when:12s} Delta {m(d['Delta'])}  "
                      f"slow decay {m(slow)}  |rotation| {m(rot)}")
        g = a["generalized"]["trained"]
        print(f"  ACTUAL generalized trained T/gamma {m(g['T_over_gamma'])}  "
              f"fast/slow residue {m(g['A']['residue_fast_over_slow_stable_modes_only'])}  "
              f"fast-component energy proxy "
              f"{m(g['A']['fast_component_energy_proxy_stable_modes_only'])} "
              f"(stable modes only: {g['A']['stable']}/{g['A']['n_modes']})")
        print(f"  ACTUAL paired trained gen/native slow decay "
              f"{m(a['paired_trained']['slow_decay_generalized_over_native'])}  "
              f"(at init {m(a['paired_init']['slow_decay_generalized_over_native'])})")
        cf = s["counterfactual_at_generalized_learned_params"]
        for v in ("A", "B", "C", "D"):
            print(f"  COUNTERFACTUAL {v}: stable {cf[v]['stable']}/{cf[v]['n_modes']} "
                  f"unstable {cf[v]['unstable']} (max radius {cf[v]['max_radius']:.6f})  "
                  f"stable-mode slow decay {m(cf[v]['slow_decay_tokens_stable_modes_only'])}  "
                  f"stable-mode DC delay {m(cf[v]['dc_delay_tokens_stable_modes_only'])}")


if __name__ == "__main__":
    try:
        main()
    except CheckpointError as err:
        print(f"checkpoint_compare: ERROR: {err}", file=sys.stderr)
        sys.exit(2)
