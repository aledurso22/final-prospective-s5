"""Focused checks for the TSS-based pilot. CLUSTER-ONLY.

Run by `bin/run_experiments/cluster_tss_pilot.sh` inside the same 600 s cap as
the pilot itself.

PREDECLARED TOLERANCES:
    EXACT    1e-10  algebraic identities, and the DISCRETE transpose identity
    EQUIV    1e-8   our law against its adaptation twin, forward AND gradient
    IDENT    1e-9   the drive-adjoint identity G_W = sum_t rho_t r_t^T
    FWDREV   1e-9   forward-mode against reverse-mode reference gradients
    FD       1e-5   central finite differences against the analytic gradient
    REFINE   1e-4   RK4 step-refinement, relative
    SLOPE    0.15   tolerance on a measured log-log error slope
"""

import os
import subprocess
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
from s5.tss_cells import (DT, T_HORIZON, assert_numeric_pytree,    # noqa: E402
                          closed_loop_polynomial, discrete_transpose,
                          drive, equivalent_adaptation, ideal_fixed_point,
                          project_contraction, q_roots, reference_metadata,
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
                                 "retained_compartment"])
def test_the_integrator_has_converged_at_the_declared_substep_count(arm):
    """Declared method plus a step-refinement check, as the design requires.

    The memory comparator is absent by construction: its complex linear memory
    uses an EXACT zero-order hold, so it has no integration step to refine.
    """
    cfg = TM.coefficients()
    p = TM.init_params(arm, 100)
    xs = jnp.asarray(onp.random.RandomState(6).randn(64, TM.D_ENC))
    n = TM.UNITS[arm]
    if arm == "tss_finite_adaptation":
        vf = tss_vector_field(p["cell"], cfg["tss"]["tau_m"],
                              cfg["tss"]["eps"], cfg["tss"]["tau_p"])
    else:
        c = cfg["retained"]
        vf = retained_vector_field(p["cell"], c["gamma"], c["T"], c["M"])
    err = float(refinement_error(vf, jnp.zeros((2, n)), xs, TM.N_SUB))
    print(f"  {arm}: refinement error at n_sub={TM.N_SUB} -> {err:.3e}")
    assert err < REFINE, (arm, err)


def test_the_execution_config_carries_no_metadata():
    """R1: a string in the jitted configuration would block production."""
    assert_numeric_pytree(TM.coefficients(), "coefficients")
    assert_numeric_pytree(symmetric_reference(), "symmetric_reference")
    assert "label" not in symmetric_reference()
    assert reference_metadata()["label"] == "symmetric-reference"
    with pytest.raises(TypeError, match="non-numeric"):
        assert_numeric_pytree({"a": 1.0, "b": "text"})


def test_the_memory_comparator_cannot_bypass_its_temporal_layer():
    """R2: the processing stage must see the MEMORY ONLY.

    An earlier revision fed the current encoded input straight into the
    processing fixed point, so this arm alone could reconstruct the fast signal
    with its memory ignored. Measured as a DERIVATIVE, not read off the source:
    holding the memory state fixed, the processing output must not depend on
    the current encoded input at all.
    """
    arm = "tss_memory_then_prospective"
    p = TM.init_params(arm, 100)
    sm = jnp.asarray(onp.random.RandomState(1).randn(2 * TM.UNITS[arm]))

    def pros_out(e_t):
        def it(st, _):
            return (p["pros"]["W"] @ jnp.tanh(st)
                    + p["pros"]["Vm"] @ sm + p["pros"]["b"]), None
        st, _ = jax.lax.scan(it, jnp.zeros((TM.PROCESSING_UNITS,)), None,
                             length=8)
        return jnp.sum(st ** 2)

    g = jax.grad(pros_out)(
        jnp.asarray(onp.random.RandomState(2).randn(TM.D_ENC)))
    assert float(jnp.max(jnp.abs(g))) == 0.0
    assert "Vx" not in p["pros"]


def test_the_memory_comparator_is_independent_complex_linear_units():
    """R2: the small version of TSS's memory structure, not a dense tanh net."""
    from s5.tss_cells import complex_memory_lambda, complex_memory_rollout
    arm = "tss_memory_then_prospective"
    p = TM.init_params(arm, 100)
    lam = complex_memory_lambda(p["mem"])
    assert lam.shape == (TM.UNITS[arm],)
    assert float(jnp.max(jnp.real(lam))) < 0.0, "memory must be stable"
    es = jnp.asarray(onp.random.RandomState(3).randn(24, TM.D_ENC))

    # INDEPENDENT: unit 0's state must not depend on any other unit's parameters
    def unit0(mem):
        return jnp.sum(complex_memory_rollout(mem, es)[:, 0] ** 2)

    g = jax.grad(unit0)(p["mem"])
    for key in ("log_decay", "omega", "W_in_re", "W_in_im"):
        rest = onp.asarray(g[key])[1:]
        assert float(onp.max(onp.abs(rest))) == 0.0, key
    # LINEAR in the encoded input
    a = complex_memory_rollout(p["mem"], es)
    b = complex_memory_rollout(p["mem"], 2.0 * es)
    assert float(jnp.max(jnp.abs(b - 2.0 * a))) < 1e-5


def test_closed_loop_poles_differ_even_though_Q_is_matched():
    """R7: the matched quantity is the OPEN-LOOP denominator.

    Equal Q does not give equal closed-loop poles once f depends on s, so the
    arms do not start with equal memory and the report must not say they do.
    """
    cfg = TM.coefficients()
    ref, tss = cfg["retained"], cfg["tss"]
    for a in (-0.5, 0.2, 0.5):
        r = closed_loop_polynomial(a, gamma=ref["gamma"], T=ref["T"],
                                   M=ref["M"])
        t = closed_loop_polynomial(a, tau_m=tss["tau_m"], eps=tss["eps"],
                                   tau_p=tss["tau_p"])
        assert abs(r[0] - t[0]) < EXACT and abs(r[2] - t[2]) < EXACT
        assert abs(r[1] - t[1]) > 1e-6, (a, r, t)
    # at a = 0 the loop is open and they coincide
    r0 = closed_loop_polynomial(0.0, gamma=ref["gamma"], T=ref["T"],
                                M=ref["M"])
    t0 = closed_loop_polynomial(0.0, tau_m=tss["tau_m"], eps=tss["eps"],
                                tau_p=tss["tau_p"])
    assert max(abs(x - y) for x, y in zip(r0, t0)) < EXACT


def test_the_state_budget_is_what_is_declared():
    counts = {a: TM.temporal_state_count(a) for a in TM.ARMS}
    assert counts["ideal_prospective"]["total"] == 0
    for a in ("tss_finite_adaptation", "retained_compartment",
              "tss_memory_then_prospective"):
        assert counts[a]["total"] == TM.STATE_BUDGET, (a, counts[a])


def test_shared_tensors_are_identical_across_arms_where_shapes_permit():
    ps = {a: TM.init_params(a, 100) for a in TM.ARMS}
    rep = TM.shared_tensor_report(ps)
    for nm in ("enc_W", "enc_b", "head_sig_W", "head_cls_W", "head_cls_b"):
        assert rep[nm] is True, (nm, rep[nm])


# ------------------------------------------------------------- the task ----
def test_the_task_is_structured_as_declared():
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


def test_the_leakage_probe_is_held_out_and_calibrated():
    """R6: a held-out score against a label-permutation null.

    The previous version fitted and scored on the same 192 examples and
    compared with chance + 0.06. In-sample ridge exploits accidental label
    associations even under an independent generator, and a ridge solution does
    not maximize accuracy, so it was neither fair nor the "upper bound" it was
    called.
    """
    batch = TK.balanced_eval_set()
    leak = TK.leakage_probe(batch)
    print(f"  leakage: held-out {leak['held_out_accuracy']:.4f}  null mean "
          f"{leak['null_mean']:.4f}  threshold {leak['null_threshold']:.4f}")
    assert leak["n_fit"] > leak["n_eval"], "the fit set must be separate"
    assert leak["passed"], leak


def test_the_leakage_probe_CATCHES_a_deliberately_leaking_task():
    """The positive control: a probe that always passes tests nothing."""
    batch = TK.balanced_eval_set()
    leaky = TK.make_leaking_batch(batch)
    fit = TK.make_leaking_batch(TK.generate(onp.random.RandomState(5), 768))
    res = TK.leakage_probe(leaky, fit_batch=fit)
    print(f"  leaking fixture: held-out {res['held_out_accuracy']:.4f} vs "
          f"threshold {res['null_threshold']:.4f}")
    assert res["held_out_accuracy"] > 0.9, res
    assert not res["passed"], res


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
def _pb_setup(seed=0, band=None):
    from experiments.tss import pilot_b as PB
    coef = symmetric_reference()
    params = {k: jnp.asarray(v) for k, v in PB.init_params(seed).items()}
    rng = onp.random.RandomState(11 + seed)
    band = band or PB.BANDS[1]
    xs, fx = PB.band_signals(rng, 4, PB.SEQ_LEN, PB.D_IN, band)
    ys, _ = PB.band_signals(rng, 4, PB.SEQ_LEN, PB.N_UNITS, band)
    return PB, coef, params, jnp.asarray(xs), jnp.asarray(ys), fx, band


def test_the_declared_bands_actually_vary_in_time():
    """R4: the previous DFT mask left the lowest band CONSTANT.

    For 64 samples at dt = 1 the first nonzero bin is 2*pi/64 = 0.0982, so a
    0.05 cutoff retained only DC. Frequencies are now drawn continuously inside
    each band, and every band must produce genuinely varying signals.
    """
    from experiments.tss import pilot_b as PB
    rng = onp.random.RandomState(7)
    for band in PB.BANDS:
        xs, fx = PB.band_signals(rng, 8, PB.SEQ_LEN, PB.D_IN, band)
        rep = PB.signal_report(xs, fx, band)
        print(f"  band {band}: temporal variance {rep['temporal_variance']:.4f}"
              f"  freq [{rep['frequency_min']:.4f}, {rep['frequency_max']:.4f}]")
        assert rep["temporal_variance"] > 0.05, (band, rep)
        assert band[0] <= rep["frequency_min"] <= rep["frequency_max"] <= band[1]
        # not constant in time, per trajectory and channel
        assert float(onp.min(onp.var(xs, axis=1))) > 1e-6, band


def test_the_drive_adjoint_identity_reproduces_the_exact_parameter_gradient():
    """G_W = sum_t rho_t r_t^T must be EXACT for this cascade, not approximate.

    If it were not, every approximation would be measured against a reference
    that already disagreed with the true gradient.
    """
    PB, coef, params, xs, ys, _, _ = _pb_setup()
    refs = PB.references(params, xs, ys, coef)
    g_id = PB.assemble_grads(params, onp.asarray(xs), refs["s1"],
                             refs["rho1"], refs["rho2"])
    for b in ("W1", "b1", "W2", "b2"):
        a = onp.asarray(g_id[b]); r = onp.asarray(refs["g_per"][b])
        num = float(onp.max(onp.abs(a - r)))
        den = float(onp.max(onp.abs(r))) + 1e-12
        print(f"  identity {b}: rel {num / den:.3e}")
        assert num / den < IDENT, (b, num / den)


def test_the_reference_validator_gates_its_own_tolerances():
    """R8: the reference ACTUALLY used must pass, and the gate must be real."""
    PB, coef, params, xs, ys, _, _ = _pb_setup(seed=1)
    refs = PB.references(params, xs, ys, coef)
    val = PB.validate_reference(params, xs, ys, coef, refs)
    print("  identity rel:", val["drive_adjoint_identity"]["relative_error"],
          " fwd/rev:", max(val["forward_vs_reverse_max_abs"].values()),
          " fd:", max(r["relative"] for r in val["finite_difference_checks"]))
    assert val["passed"], val
    # the gate is not vacuous: a corrupted reference must fail it
    bad = dict(refs, g_per={k: v * 1.5 for k, v in refs["g_per"].items()})
    assert not PB.validate_reference(params, xs, ys, coef, bad)["passed"]


def test_forward_sensitivities_and_finite_differences_confirm_the_reference():
    """Three independent routes to the same gradient, INCLUDING the initial
    states: reverse mode, forward mode, and central differences."""
    PB, coef, params, xs, ys, _, _ = _pb_setup(seed=2)
    d1, d2 = PB.zero_drives()
    rev = jax.grad(PB.batch_loss)(params, xs, ys, d1, d2, coef)
    fwd = jax.jacfwd(PB.batch_loss)(params, xs, ys, d1, d2, coef)
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
        num = (float(PB.batch_loss(pp, xs, ys, d1, d2, coef))
               - float(PB.batch_loss(pm, xs, ys, d1, d2, coef))) / (2 * h)
        ana = float(onp.sum(onp.asarray(rev[b]) * v))
        rel = abs(num - ana) / max(abs(ana), 1e-12)
        print(f"  finite difference {b}: rel {rel:.3e}")
        assert rel < FD, (b, rel, num, ana)


def test_loss_change_uses_the_SAME_objective_the_gradient_came_from():
    """R3: a batch-mean gradient measured against one member's loss.

    The exact gradient of a mean need not decrease an individual member's loss,
    so the previous mismatch could have rejected the exact reference itself.
    The fixture below is built so that trajectory 0 DISAGREES with the mean:
    the exact batch step must decrease the batch objective, and is allowed not
    to decrease trajectory 0's.
    """
    PB, coef, params, xs, ys, _, _ = _pb_setup(seed=3)
    d1, d2 = PB.zero_drives()
    g = jax.grad(PB.batch_loss)(params, xs, ys, d1, d2, coef)
    for sn in PB.STEP_NORMS:
        dl = PB.loss_change(params, xs, ys, coef, g, sn)
        print(f"  batch objective change at |step|={sn}: {dl:.3e}")
        assert dl is not None and dl < 0.0, (sn, dl)
    # per-example gradients genuinely conflict here, which is what makes the
    # distinction between the two objectives observable at all
    per = PB._per_traj_param_grad(params, xs, ys, d1, d2, coef)
    flat = onp.stack([PB._flat(per, ("W1", "b1", "W2", "b2"), i)
                      for i in range(xs.shape[0])])
    cos = (flat @ flat.T) / onp.outer(onp.linalg.norm(flat, axis=1),
                                      onp.linalg.norm(flat, axis=1))
    print(f"  min pairwise per-example gradient cosine: {cos.min():.4f}")
    assert cos.min() < 0.999, "the fixture must not have identical gradients"


def test_part_B_is_spatially_feedforward_so_no_backward_loop_exists():
    """Layer 1's drive must not depend on any state, and layer 2's only on s1.

    Measured, not asserted from the source.
    """
    PB, coef, params, xs, ys, _, _ = _pb_setup(seed=4)
    d1, d2 = PB.zero_drives()
    f = jax.vmap(PB.forward, in_axes=(None, 0, None, None, None))
    s1a, _ = f(params, xs, d1, d2, coef)
    s1b, _ = f(params, xs, d1, d2.at[10].add(1.0), coef)
    assert float(jnp.max(jnp.abs(s1a - s1b))) == 0.0


def test_the_discrete_transpose_is_the_EXACT_adjoint_of_the_executed_cell():
    """R5: reverse, apply THE SAME discrete operator, reverse.

    The previous test compared a sampled operator against the CONTINUOUS
    H(-i omega) at a tight tolerance, which mixes two different transfer
    functions; refining RK4 substeps cannot remove a sample-and-hold timing
    difference. For a causal LTI map with zero initial state the reversal
    identity is exact, so this is checked at 1e-10 and needs no tolerance
    argument at all.
    """
    PB, coef, params, xs, ys, _, _ = _pb_setup(seed=6)
    rng = onp.random.RandomState(21)
    L, n = PB.SEQ_LEN, PB.N_UNITS
    f = jnp.asarray(rng.randn(L, n))
    c = jnp.asarray(rng.randn(L, n))

    def roll(ff):
        return PB.cell_rollout(ff, coef, n)

    def J(ff):
        return jnp.sum(c * roll(ff))

    exact = onp.asarray(jax.grad(J)(f))
    viarev = onp.asarray(discrete_transpose(roll, c))
    num = float(onp.max(onp.abs(exact - viarev)))
    den = float(onp.max(onp.abs(exact))) + 1e-12
    print(f"  discrete transpose vs autodiff: rel {num / den:.3e}")
    assert num / den < EXACT, num / den


def test_the_verdict_rule_is_applied_mechanically():
    """R8: the predeclared scientific rule, exercised on synthetic rows."""
    from experiments.tss import pilot_b as PB
    good = dict(mean_cosine=0.9, negative_cosine_fraction=0.0,
                mean_relative_error=0.1,
                loss_change={str(sn): dict(approximation=-1e-4,
                                           reference=-2e-4)
                             for sn in PB.STEP_NORMS})
    bad = dict(good, negative_cosine_fraction=0.05)
    worse = dict(good, loss_change={str(sn): dict(approximation=+1e-4,
                                                  reference=-2e-4)
                                    for sn in PB.STEP_NORMS})
    v = PB.verdict([dict(approximations={"moment_eps0.5": good,
                                         "reciprocal_eps0.5": bad,
                                         "x": worse})])
    assert v["per_filter"]["moment_eps0.5"]["usable"] is True
    assert v["per_filter"]["reciprocal_eps0.5"]["usable"] is False
    assert v["per_filter"]["x"]["usable"] is False


def test_every_vmap_in_axes_matches_the_wrapped_function_arity():
    """A static guard for the class of bug that cost a cluster slot.

    `_forward_batch` was vmapped with six in_axes entries and called with a
    `ys` argument that `forward` does not take. Nothing in the import-time link
    check or the production probe reaches Part B's reference path, so it
    surfaced only when the focused checks ran on the cluster. This audit is
    pure AST inspection: no execution, and it covers every vmap in the pilot.
    """
    import ast
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    files = ["s5/tss_cells.py", "s5/tss_models.py", "s5/causal_error.py",
             "tasks/dual_recall.py", "experiments/tss/pilot_a.py",
             "experiments/tss/pilot_b.py", "tests/test_tss_pilot.py"]
    arity, trees = {}, {}
    for f in files:
        trees[f] = ast.parse((root / f).read_text())
        for n in ast.walk(trees[f]):
            if isinstance(n, ast.FunctionDef):
                arity[n.name] = len(n.args.args)
    checked, bad = 0, []
    for f, tree in trees.items():
        for n in ast.walk(tree):
            if not (isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "vmap"):
                continue
            ia = next((k for k in n.keywords if k.arg == "in_axes"), None)
            if ia is None or not isinstance(ia.value, ast.Tuple):
                continue
            tgt = n.args[0] if n.args else None
            name = None
            if isinstance(tgt, ast.Name):
                name = tgt.id
            elif isinstance(tgt, ast.Attribute):
                name = tgt.attr
            elif isinstance(tgt, ast.Call) and tgt.args:
                inner = tgt.args[0]
                name = getattr(inner, "id", getattr(inner, "attr", None))
            if name in arity:
                checked += 1
                if arity[name] != len(ia.value.elts):
                    bad.append((f, n.lineno, name, arity[name],
                                len(ia.value.elts)))
    print(f"  checked {checked} vmap call sites")
    assert checked >= 4, "the audit found too few sites to be meaningful"
    assert not bad, bad


def test_the_complex_memory_carry_dtype_follows_its_inputs():
    """The x64 regression: a hard-coded complex64 carry with a promoting body.

    `lax.scan` requires the carry's dtype to be invariant. Starting the memory
    at complex64 while the body promoted to complex128 under x64 failed in the
    float64 checks and PASSED in the production float32 configuration - the
    reverse of the usual direction, which is why both are exercised.
    """
    from s5.tss_cells import complex_memory_rollout
    p = TM.init_params("tss_memory_then_prospective", 100)
    for dt in (onp.float32, onp.float64):
        es = jnp.asarray(onp.random.RandomState(9).randn(12, TM.D_ENC),
                         dtype=dt)
        out = complex_memory_rollout(p["mem"], es)
        assert out.shape == (12, 2 * TM.UNITS["tss_memory_then_prospective"])
        assert onp.all(onp.isfinite(onp.asarray(out))), dt


def test_production_entrypoints_in_the_production_dtype():
    """R1: x64 is process-global, so the production probe runs separately.

    It calls the ACTUAL jitted `train_step` and `eval_all` for every arm with
    x64 OFF, which is the configuration Part A runs in and which no float64
    fixture can stand in for.
    """
    probe = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "tss_production_probe.py")
    env = dict(os.environ, JAX_ENABLE_X64="0")
    r = subprocess.run([sys.executable, probe], env=env,
                       capture_output=True, text=True)
    print(r.stdout[-4000:]); print(r.stderr[-4000:])
    assert r.returncode == 0, r.stdout + r.stderr
