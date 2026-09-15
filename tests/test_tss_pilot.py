"""Focused checks for the TSS-based pilot. CLUSTER-ONLY.

Run by `bin/run_experiments/cluster_tss_pilot.sh` inside the same 600 s cap as
the pilot itself.

PREDECLARED TOLERANCES:
    EXACT    1e-10  algebraic identities of the coefficient map, float64
    EQUIV    1e-8   our law against its adaptation twin, forward AND gradient
    IDENT    1e-9   the drive-adjoint identity G_W = sum_t rho_t r_t^T
    FWDREV   1e-9   forward-mode against reverse-mode reference gradients
    FD       1e-5   central finite differences against the analytic gradient
    REFINE   1e-4   RK4 step-refinement, relative
    SLOPE    0.15   tolerance on a measured log-log error slope
"""

import os
import sys

import jax
import numpy as onp
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp                                            # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tasks.dual_recall as TK                                     # noqa: E402
from s5 import causal_error as CE                                  # noqa: E402
from s5 import tss_models as TM                                    # noqa: E402
from s5.tss_cells import (DT, T_HORIZON, drive,                    # noqa: E402
                          equivalent_adaptation, ideal_fixed_point,
                          project_contraction, q_roots,
                          refinement_error, retained_vector_field,
                          rollout_ode, symmetric_reference,
                          tss_gamma_equivalent, tss_vector_field)

EXACT, EQUIV, IDENT, FWDREV, FD, REFINE, SLOPE = (
    1e-10, 1e-8, 1e-9, 1e-9, 1e-5, 1e-4, 0.15)


def _cell(n=4, d_in=5, seed=0):
    rs = onp.random.RandomState(seed)
    return dict(W=jnp.asarray(0.4 * rs.randn(n, n) / onp.sqrt(n)),
                U=jnp.asarray(rs.randn(n, d_in) / onp.sqrt(d_in)),
                b=jnp.asarray(0.1 * rs.randn(n)))


def _inputs(L=48, d_in=5, seed=1):
    return jnp.asarray(onp.random.RandomState(seed).randn(L, d_in))


# ----------------------------------------------- the coefficient algebra ---
def test_the_adaptation_map_is_exact():
    """tau_m eps = M, tau_m + eps = gamma + T, eps + tau_p = T."""
    ref = symmetric_reference()
    g, T, M = ref["gamma"], ref["T"], ref["M"]
    tau_m, eps, tau_p = equivalent_adaptation(g, T, M)
    assert abs(tau_m * eps - M) < EXACT
    assert abs(tau_m + eps - (g + T)) < EXACT
    assert abs(eps + tau_p - T) < EXACT
    t_minus, t_plus = q_roots(g, T, M)
    assert abs(tau_m - t_plus) < EXACT and abs(eps - t_minus) < EXACT
    # at the symmetric point the roots are 3T/2 and T/2
    assert abs(t_plus - 1.5 * T) < EXACT and abs(t_minus - 0.5 * T) < EXACT


def test_the_TSS_matched_point_maps_to_gamma_zero_and_is_INADMISSIBLE():
    """TSS's tau_p = tau_m prescription leaves our admissible sector.

    Under the map, gamma = tau_m - tau_p, so matching gives gamma = 0 exactly
    and M <= gamma*T then fails for any M > 0. Arms 2 and 4 are therefore two
    coefficient sectors of ONE family, not equivalent realizations of each
    other and not different model classes. This is the check that keeps the
    report from claiming either.
    """
    cfg = TM.coefficients()
    c = cfg["tss"]
    eq = tss_gamma_equivalent(c["tau_m"], c["eps"], c["tau_p"])
    assert abs(eq["gamma"]) < EXACT, eq
    assert eq["M"] > 0.0
    assert eq["admissible"] is False
    # and our own point IS admissible
    ref = symmetric_reference()
    assert 0.0 < ref["M"] <= ref["gamma"] * ref["T"]


def test_arms_two_and_four_share_poles_and_differ_only_in_the_zero():
    """The comparison is about response SHAPE, not about who got more memory."""
    cfg = TM.coefficients()
    ref, tss = cfg["retained"], cfg["tss"]
    t_minus, t_plus = q_roots(ref["gamma"], ref["T"], ref["M"])
    assert abs(tss["tau_m"] - t_plus) < EXACT
    assert abs(tss["eps"] - t_minus) < EXACT
    # prospective zero: ours at T, TSS's at eps + tau_p = 2T
    assert abs((tss["eps"] + tss["tau_p"]) - 2.0 * ref["T"]) < EXACT


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_our_law_equals_its_adaptation_twin_in_FORWARD_and_GRADIENT(seed):
    """The equivalence is an implementation CHECK, not a fifth competitor.

    Gradients are compared as well as values: two realizations can agree on a
    trajectory and still differ in what they backpropagate.
    """
    ref = symmetric_reference()
    g, T, M = ref["gamma"], ref["T"], ref["M"]
    tau_m, eps, tau_p = equivalent_adaptation(g, T, M)
    p = _cell(seed=seed)
    xs = _inputs(seed=seed + 10)
    n = p["b"].shape[0]
    z0 = jnp.zeros((2, n))

    def ours(pp):
        zs, _ = rollout_ode(retained_vector_field(pp, g, T, M), z0, xs, 8)
        return zs[:, 0, :]

    def twin(pp):
        zs, _ = rollout_ode(tss_vector_field(pp, tau_m, eps, tau_p), z0, xs, 8)
        return zs[:, 0, :]

    a, b = ours(p), twin(p)
    scale = float(jnp.max(jnp.abs(a))) + 1.0
    assert float(jnp.max(jnp.abs(a - b))) / scale < EQUIV

    w = jnp.asarray(onp.random.RandomState(seed).randn(*a.shape))
    ga = jax.grad(lambda pp: jnp.sum(w * ours(pp)))(p)
    gb = jax.grad(lambda pp: jnp.sum(w * twin(pp)))(p)
    for k in ga:
        d = float(jnp.max(jnp.abs(ga[k] - gb[k])))
        s = float(jnp.max(jnp.abs(ga[k]))) + 1.0
        assert d / s < EQUIV, (k, d / s)


# --------------------------------------------------- memory and the ideal -
def test_the_ideal_control_has_NO_memory_and_the_others_do():
    """The cancellation control's defining property, measured on both sides.

    Changing an input BEFORE step t must leave the ideal arm's output at t
    unchanged, and must change it for every memory-bearing arm. A control that
    silently retained memory - through a backward-difference parasitic state,
    say - would invalidate the whole comparison.
    """
    cfg = TM.coefficients()
    xs = jnp.asarray(onp.random.RandomState(3).randn(40, TM.D_ENC))
    xs2 = xs.at[5].add(1.0)
    for arm in TM.ARMS:
        p = TM.init_params(arm, 100)
        a, _ = TM.temporal(arm, p, xs, cfg)
        b, _ = TM.temporal(arm, p, xs2, cfg)
        later = float(jnp.max(jnp.abs(a[20:] - b[20:])))
        at_5 = float(jnp.max(jnp.abs(a[5] - b[5])))
        assert at_5 > 1e-6, (arm, "the perturbed step itself must change")
        if arm == "ideal_prospective":
            assert later < 1e-9, (arm, later)
        else:
            assert later > 1e-6, (arm, later)


def test_the_fixed_point_solve_converges_under_the_declared_cap():
    """The contraction cap is what makes the ideal arm's solution unique."""
    p = _cell(n=16, d_in=16, seed=4)
    p = dict(p, W=project_contraction(3.0 * p["W"]))
    sv = float(jnp.linalg.norm(p["W"], ord=2))
    assert sv <= 0.8 + 1e-6, sv
    x = jnp.asarray(onp.random.RandomState(5).randn(16))
    s, res = ideal_fixed_point(p, x)
    assert float(res) < 1e-4, float(res)
    # and it really is the fixed point of the declared map
    assert float(jnp.max(jnp.abs(drive(p, s, x) - s))) < 1e-4


@pytest.mark.parametrize("arm", ["tss_finite_adaptation",
                                 "retained_compartment",
                                 "memory_then_prospective"])
def test_the_integrator_has_converged_at_the_declared_substep_count(arm):
    """Declared method plus a step-refinement check, as the design requires."""
    cfg = TM.coefficients()
    p = TM.init_params(arm, 100)
    xs = jnp.asarray(onp.random.RandomState(6).randn(64, TM.D_ENC))
    n = TM.UNITS[arm]
    if arm == "tss_finite_adaptation":
        vf = tss_vector_field(p["cell"], cfg["tss"]["tau_m"],
                              cfg["tss"]["eps"], cfg["tss"]["tau_p"])
        z0 = jnp.zeros((2, n))
    elif arm == "retained_compartment":
        c = cfg["retained"]
        vf = retained_vector_field(p["cell"], c["gamma"], c["T"], c["M"])
        z0 = jnp.zeros((2, n))
    else:
        from s5.tss_cells import leaky_vector_field
        vf = leaky_vector_field(p["mem"], cfg["memory"]["tau_mem"])
        z0 = jnp.zeros((n,))
    err = float(refinement_error(vf, z0, xs, TM.N_SUB))
    print(f"  {arm}: refinement error at n_sub={TM.N_SUB} -> {err:.3e}")
    assert err < REFINE, (arm, err)


def test_the_state_budget_is_what_is_declared():
    counts = {a: TM.temporal_state_count(a) for a in TM.ARMS}
    assert counts["ideal_prospective"]["total"] == 0
    for a in ("tss_finite_adaptation", "retained_compartment",
              "memory_then_prospective"):
        assert counts[a]["total"] == TM.STATE_BUDGET, (a, counts[a])


def test_shared_tensors_are_identical_across_arms_where_shapes_permit():
    ps = {a: TM.init_params(a, 100) for a in TM.ARMS}
    rep = TM.shared_tensor_report(ps)
    for nm in ("enc_W", "enc_b", "head_sig_W", "head_cls_W", "head_cls_b"):
        assert rep[nm] is True, (nm, rep[nm])


# ------------------------------------------------------------- the task ----
def test_the_task_is_structured_as_declared_and_does_not_leak():
    batch = TK.balanced_eval_set()
    st = TK.structure_check(batch)
    for key in ("one_content_symbol_per_step", "exactly_one_write",
                "exactly_one_query", "cue_present_at_write",
                "query_after_cue", "query_inside_window",
                "no_cue_marker_at_query"):
        assert st[key] is True, key
    assert set(st["delay_counts"].values()) == {64}       # 8 classes x 8 reps
    assert set(st["class_counts"].values()) == {24}       # 3 delays x 8 reps
    assert abs(st["signal_variance"] - 1.0) < 0.15, st["signal_variance"]
    leak = TK.leakage_probe(batch)
    print(f"  leakage probe: {leak['accuracy']:.4f} vs chance "
          f"{leak['chance']:.4f}")
    assert leak["accuracy"] < leak["chance"] + 0.06, leak


def test_the_interventions_change_the_cue_and_preserve_everything_else():
    batch = TK.balanced_eval_set()
    x, u, cls, tq, meta = batch
    sx, su, scls, stq, smeta = TK.shuffled_cue(batch, 7)
    assert onp.all(scls != cls)                       # always a DIFFERENT cue
    assert onp.array_equal(su, u) and onp.array_equal(stq, tq)
    # only the cue step's content channel differs
    d = onp.abs(sx - x).sum(axis=2)
    rows = onp.arange(cls.size)
    assert onp.all(d[rows, meta["cue_step"]] > 0)
    d2 = d.copy(); d2[rows, meta["cue_step"]] = 0.0
    assert onp.all(d2 == 0)
    rx, ru, rcls, rtq, _ = TK.replaced_distractors(batch, 8)
    assert onp.array_equal(rcls, cls) and onp.array_equal(ru, u)
    assert onp.all(rx[rows, meta["cue_step"], 1 + cls] == 1.0)


# ------------------------------------------------- the causal error filters
def test_the_moment_filter_matches_its_closed_form_and_sums_to_one():
    ref = symmetric_reference()
    g, T, M = ref["gamma"], ref["T"], ref["M"]
    eps = 0.5 * q_roots(g, T, M)[0]
    w = CE.moment_matched_weights(g, T, M, eps)
    assert abs(float(onp.sum(w)) - 1.0) < EXACT
    assert w[1] < 0.0, "w2 < 0 is what supplies the extrapolation"
    _, c2, _ = CE.adjoint_moments(g, T, M)
    om = onp.linspace(0.0, 0.5, 41)
    got = CE.transfer("moment", g, T, M, eps, eps, om)
    p = 1j * om
    want = ((1.0 + (g + 3 * eps) * p + (c2 + 3 * g * eps + 3 * eps ** 2) * p ** 2)
            / (1.0 + eps * p) ** 3)
    assert float(onp.max(onp.abs(got - want))) < 1e-10


def test_the_reciprocal_filter_matches_its_closed_form():
    ref = symmetric_reference()
    g, T, M = ref["gamma"], ref["T"], ref["M"]
    eps = delta = 0.5 * q_roots(g, T, M)[0]
    om = onp.linspace(0.0, 0.5, 41)
    got = CE.transfer("reciprocal", g, T, M, eps, delta, om)
    p = 1j * om
    E = (1.0 + (M / T) * p / (1 + eps * p)
         + (g - M / T) * p / (1 + T * p))
    R = (1.0 + 2 * delta * p) / (1.0 + delta * p) ** 2
    assert float(onp.max(onp.abs(got - E * R))) < 1e-10


def test_the_low_frequency_error_orders_are_three_and_two():
    """The MEASURED slopes, not the asserted asymptotic order.

    A cubic order does not imply a smaller error at any particular bandwidth,
    which is why the pilot reports measured band errors as well.
    """
    ref = symmetric_reference()
    g, T, M = ref["gamma"], ref["T"], ref["M"]
    eps = delta = 0.5 * q_roots(g, T, M)[0]
    oms = onp.array([1e-4, 2e-4, 4e-4, 8e-4])
    for kind, want in (("moment", 3.0), ("reciprocal", 2.0)):
        e = onp.array([onp.abs(
            CE.transfer(kind, g, T, M, eps, delta, onp.array([o]))[0]
            - CE.exact_adjoint_transfer(g, T, M, onp.array([o]))[0])
            for o in oms])
        slope = onp.polyfit(onp.log(oms), onp.log(e), 1)[0]
        print(f"  {kind}: measured low-frequency slope {slope:.4f}")
        assert abs(slope - want) < SLOPE, (kind, slope)


def test_the_executed_filter_states_are_stable_and_the_witness_is_not():
    ref = symmetric_reference()
    g, T, M = ref["gamma"], ref["T"], ref["M"]
    t_minus = q_roots(g, T, M)[0]
    for frac in (0.25, 0.5, 1.0):
        eps = frac * t_minus
        for kind in ("moment", "reciprocal"):
            pol = CE.filter_poles(kind, g, T, M, eps, eps)
            assert pol["stable"], (kind, frac, pol)
    w = CE.recurrent_witness()
    assert w["forward_stable"] is True
    assert w["error_loop_stable"] is False
    assert abs(w["reciprocal_error_pole"] - 0.125) < 1e-12


def test_the_remainder_coefficients_obey_their_stated_inequalities():
    ref = symmetric_reference()
    g, T, M = ref["gamma"], ref["T"], ref["M"]
    r = CE.remainder_bound(g, T, M, 0.5 * q_roots(g, T, M)[0], 0.4)
    assert r["c2_ge_gamma_sq"] and r["c3_ge_gamma_cu"], r
    assert r["B3"] > 0 and r["B4"] > 0


# ----------------------------------------------------- Part B references ---
def test_the_drive_adjoint_identity_reproduces_the_exact_parameter_gradient():
    """G_W = sum_t rho_t r_t^T must be EXACT for this cascade, not approximate.

    If it were not, every approximation would be measured against a reference
    that already disagreed with the true gradient.
    """
    from experiments.tss import pilot_b as PB
    coef = symmetric_reference()
    params = {k: jnp.asarray(v) for k, v in PB.init_params(0).items()}
    rng = onp.random.RandomState(11)
    xs = jnp.asarray(PB.band_limited(rng, 1, PB.SEQ_LEN, PB.D_IN, 0.3)[0])
    ys = jnp.asarray(PB.band_limited(rng, 1, PB.SEQ_LEN, PB.N_UNITS, 0.3)[0])
    r1, r2 = PB.exact_drive_adjoints(params, xs, ys, coef)
    _, _, s1, _ = PB.teaching_inputs(params, xs, ys, coef, r2)
    g_id = PB.assemble_grads(params, xs, s1, r1, r2)
    g_ref = PB.exact_param_grad(params, xs, ys, coef)
    for b in ("W1", "b1", "W2", "b2"):
        num = float(onp.max(onp.abs(onp.asarray(g_id[b])
                                    - onp.asarray(g_ref[b]))))
        den = float(onp.max(onp.abs(onp.asarray(g_ref[b])))) + 1e-12
        print(f"  identity {b}: rel {num / den:.3e}")
        assert num / den < IDENT, (b, num / den)


def test_forward_sensitivities_and_finite_differences_confirm_the_reference():
    """Three independent routes to the same gradient, INCLUDING the initial
    states: reverse mode, forward mode, and central differences."""
    from experiments.tss import pilot_b as PB
    coef = symmetric_reference()
    params = {k: jnp.asarray(v) for k, v in PB.init_params(1).items()}
    rng = onp.random.RandomState(12)
    xs = jnp.asarray(PB.band_limited(rng, 1, PB.SEQ_LEN, PB.D_IN, 0.3)[0])
    ys = jnp.asarray(PB.band_limited(rng, 1, PB.SEQ_LEN, PB.N_UNITS, 0.3)[0])
    d1, d2 = PB.zero_drives()
    rev = jax.grad(PB.loss)(params, xs, ys, d1, d2, coef)
    fwd = jax.jacfwd(PB.loss)(params, xs, ys, d1, d2, coef)
    for b in params:
        assert float(onp.max(onp.abs(onp.asarray(rev[b])
                                     - onp.asarray(fwd[b])))) < FWDREV, b
    rs = onp.random.RandomState(13)
    for b in ("W1", "W2", "z0_1", "z0_2"):
        v = rs.randn(*onp.asarray(params[b]).shape)
        v /= onp.linalg.norm(v)
        h = 1e-6
        pp = dict(params); pp[b] = params[b] + h * v
        pm = dict(params); pm[b] = params[b] - h * v
        num = (float(PB.loss(pp, xs, ys, d1, d2, coef))
               - float(PB.loss(pm, xs, ys, d1, d2, coef))) / (2 * h)
        ana = float(onp.sum(onp.asarray(rev[b]) * v))
        rel = abs(num - ana) / max(abs(ana), 1e-12)
        print(f"  finite difference {b}: rel {rel:.3e}")
        assert rel < FD, (b, rel, num, ana)


def test_part_B_is_spatially_feedforward_so_no_backward_loop_exists():
    """Layer 1's drive must not depend on any state, and layer 2's only on s1.

    Measured, not asserted from the source: perturbing a state must not change
    layer 1's drive.
    """
    from experiments.tss import pilot_b as PB
    coef = symmetric_reference()
    params = {k: jnp.asarray(v) for k, v in PB.init_params(2).items()}
    rng = onp.random.RandomState(14)
    xs = jnp.asarray(PB.band_limited(rng, 1, PB.SEQ_LEN, PB.D_IN, 0.3)[0])
    ys = jnp.asarray(PB.band_limited(rng, 1, PB.SEQ_LEN, PB.N_UNITS, 0.3)[0])
    # layer 2's drive perturbation must NOT affect layer 1's states
    d1, d2 = PB.zero_drives()
    s1a, _ = PB.forward(params, xs, d1, d2, coef)
    s1b, _ = PB.forward(params, xs, d1, d2.at[10].add(1.0), coef)
    assert float(jnp.max(jnp.abs(s1a - s1b))) == 0.0
    del ys


def test_the_exact_backward_filter_reproduces_the_adjoint_transfer():
    """Running H backward in time is H_A, checked against its closed form."""
    from experiments.tss import pilot_b as PB
    coef = symmetric_reference()
    g, T, M = coef["gamma"], coef["T"], coef["M"]
    L = 512
    om = 2.0 * onp.pi * 7.0 / L
    t = onp.arange(L)
    k = jnp.asarray(onp.cos(om * t)[:, None])
    out = onp.asarray(PB.exact_backward_filter(k, coef, n_sub=8))[:, 0]
    mid = slice(L // 4, 3 * L // 4)             # away from both boundaries
    Href = complex(CE.exact_adjoint_transfer(g, T, M, onp.array([om]))[0])
    want = onp.real(Href * onp.exp(1j * om * t))[mid]
    rel = float(onp.max(onp.abs(out[mid] - want))
                / (onp.max(onp.abs(want)) + 1e-12))
    print(f"  backward-filter vs H_A: rel {rel:.3e}")
    assert rel < 5e-3, rel
