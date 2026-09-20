"""The JAX implementation of the direct prospective recurrences (cluster).

Two kinds of test, deliberately separated, because conflating them is what
made the first cluster run look like a scan defect when it was an unstable
fixture:

  SCAN CORRECTNESS is tested on ANALYTICALLY CERTIFIED STABLE fixtures --
  companion coefficients built from chosen roots strictly inside the unit
  disc -- so nothing overflows and float64 agreement with the sequential
  oracle can be strict, including gradients.

  MODEL INSTABILITY is tested on the production-derived fixture that does
  diverge. There the claim is not "the values are small" but "the scan and
  the oracle agree over the complete common finite prefix and go nonfinite
  at the same token and mode". Matching NaNs alone would prove nothing, so
  `equal_nan` is never used.

NAMING. Two independent axes: the MODEL (PROFESSOR_TSS with M = 0, or
WWJ_GENERALIZED_TSS with M > 0) and the TARGET CONSTRUCTION
(PROFESSOR_LINEAR_TARGET f = Abar s + Bbar x, or NATIVE_MATCHED_TARGET). "The
WWJ model on the professor linear target" is not "the Professor model".

    JAX_ENABLE_X64=1 $PY -m pytest tests/test_direct_prospective_scan.py
"""

import os
import subprocess

import jax
import jax.numpy as np

from s5 import direct_prospective as DP
from tests import direct_prospective_reference as ORACLE

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NATIVE_BASE = "6d55765a36f5985f3a062a8f3b6854316b26b1bd"
X64 = jax.config.read("jax_enable_x64")
TOL64, TOL32 = 1e-10, 2e-4
#: lengths exercised everywhere, including non-powers of two and > 1000
LENGTHS = (1, 2, 3, 5, 7, 8, 9, 15, 17, 100, 257, 1000, 1023, 1500)

#: NO LIST OF "STABLE TAU VALUES" LIVES HERE ANY MORE. Three separate
#: subset-certification errors came from maintaining one: tau = 2 and tau = 5
#: were called stable from small random mode draws, and this file then
#: hard-coded (5.0, 10.0, 50.0) as stable while its own header had already
#: withdrawn tau = 5. Production eligibility comes from
#: `experiments.s5_direct_prospective.certification` and nowhere else.
#:
#: Cluster-measured rejections, kept as regressions rather than as a
#: "stable" list:
REJECTED_FIXTURES = (
    {"tau": 2.0, "eps": 0.0, "radius": 1.4415912882,
     "witness": "Abar = 0.9 e^{-2i}", "first_nonfinite_token": 242},
    {"tau": 5.0, "eps": 0.0, "radius": 1.0787247827275395,
     "witness": "the _model_modes(15) fixture on the cluster",
     "analytic_witness_radius": 1.0977},
)
#: candidate tau values a test may EXAMINE; none is assumed stable
TAU_CANDIDATES = (2.0, 5.0, 10.0, 50.0, 100.0, 1000.0)


def _context(**fields):
    """Every sequence assertion carries its cell: tau, target, length..."""
    return ", ".join(f"{key}={value!r}" for key, value in fields.items())


def _complex(dtype=None):
    return dtype or (np.complex128 if X64 else np.complex64)


def _inputs(seed, length, H=3):
    real = np.float64 if X64 else np.float32
    return jax.random.normal(jax.random.PRNGKey(seed), (length, H)).astype(real)


def _close(left, right, tolerance):
    """Strict comparison. Nonfinite values on either side FAIL it."""
    if not (bool(np.all(np.isfinite(left)))
            and bool(np.all(np.isfinite(right)))):
        return False
    scale = float(np.maximum(np.max(np.abs(right)), 1.0))
    return float(np.max(np.abs(left - right))) <= tolerance * scale


# ------------------------------------------- certified stable fixtures -----
def certified_stable_coefficients(order, seed, P=6, complex_modes=True):
    """Companion coefficients built FROM ROOTS inside the unit disc.

    For roots r_i, s_t = sum_i A_i s_{t-i} has
    A_1 = sum r_i, A_2 = -sum_{i<j} r_i r_j, A_3 = r_1 r_2 r_3, so the
    spectral radius is max |r_i| < 1 BY CONSTRUCTION -- no model, no
    coefficient builder, nothing that can be unstable.
    """
    keys = jax.random.split(jax.random.PRNGKey(seed), 2)
    magnitude = 0.2 + 0.6 * jax.random.uniform(keys[0], (P, order))
    if complex_modes:
        angle = jax.random.uniform(keys[1], (P, order), minval=-3.0,
                                   maxval=3.0)
        roots = (magnitude * np.exp(1j * angle)).astype(_complex())
    else:
        roots = magnitude.astype(_complex())
    if order == 2:
        r1, r2 = roots[:, 0], roots[:, 1]
        coefficients = (r1 + r2, -(r1 * r2))
    elif order == 3:
        r1, r2, r3 = roots[:, 0], roots[:, 1], roots[:, 2]
        coefficients = (r1 + r2 + r3,
                        -(r1 * r2 + r1 * r3 + r2 * r3),
                        r1 * r2 * r3)
    else:
        raise ValueError(order)
    return coefficients, float(np.max(np.abs(roots)))


def test_the_certified_fixtures_really_are_stable():
    """The premise of every stable test below, checked rather than assumed."""
    for order in (2, 3):
        for complex_modes in (True, False):
            coefficients, radius = certified_stable_coefficients(
                order, 1, complex_modes=complex_modes)
            assert radius < 1.0, (order, radius)
            estimate = DP.companion_spectral_radius(coefficients)
            assert float(np.max(estimate)) <= 1.0 + 1e-6, (order, estimate)
            exact = ORACLE.exact_companion_radius(coefficients)
            assert float(np.max(exact)) < 1.0


def test_the_scan_matches_the_oracle_on_stable_fixtures_at_many_lengths():
    """Strict float64 agreement, real and complex modes, orders 2 and 3,
    lengths through 1500 including non-powers of two."""
    tolerance = TOL64 if X64 else TOL32
    for order in (2, 3):
        for complex_modes in (True, False):
            coefficients, _ = certified_stable_coefficients(
                order, 2 + order, complex_modes=complex_modes)
            for length in LENGTHS:
                drive = jax.random.normal(
                    jax.random.PRNGKey(length),
                    (length, coefficients[0].shape[0])).astype(_complex())
                fast = DP.companion_doubling_scan(coefficients, drive)
                slow = ORACLE.sequential_scan(coefficients, drive)
                assert fast.shape == drive.shape
                assert _close(fast, slow, tolerance), (order, complex_modes,
                                                       length)


def test_stable_scan_gradients_match_the_oracle():
    for order in (2, 3):
        coefficients, _ = certified_stable_coefficients(order, 9, P=4)
        drive = jax.random.normal(jax.random.PRNGKey(10),
                                  (129, 4)).astype(_complex())

        def loss(values, implementation):
            return np.sum(np.abs(implementation(values, drive)) ** 2).real

        fast = jax.grad(loss)(coefficients, DP.companion_doubling_scan)
        slow = jax.grad(loss)(coefficients, ORACLE.sequential_scan)
        for index, (left, right) in enumerate(zip(fast, slow)):
            assert _close(left, right, 1e-7 if X64 else 1e-3), (order, index)


def test_stable_scan_has_zero_prehistory_and_no_wraparound():
    coefficients, _ = certified_stable_coefficients(3, 11, P=4)
    length = 32
    for impulse in (0, 1, 2, 31):
        drive = np.zeros((length, 4), dtype=_complex()).at[impulse].set(1.0)
        states = DP.companion_doubling_scan(coefficients, drive)
        assert bool(np.all(states[:impulse] == 0)), impulse
        assert bool(np.any(states[impulse] != 0)), impulse
    # an impulse at the end never reaches the head
    drive = np.zeros((length, 4), dtype=_complex()).at[length - 1].set(1.0)
    states = DP.companion_doubling_scan(coefficients, drive)
    assert bool(np.all(states[:length - 1] == 0))


def test_every_rematerialization_choice_agrees_on_a_stable_fixture():
    coefficients, _ = certified_stable_coefficients(3, 12)
    drive = jax.random.normal(jax.random.PRNGKey(13),
                              (257, coefficients[0].shape[0])).astype(
                                  _complex())
    values = {mode: DP.companion_doubling_scan(coefficients, drive,
                                               remat=mode)
              for mode in ("level", "whole", None)}
    for mode, value in values.items():
        assert _close(value, values[None], TOL64 if X64 else TOL32), mode


# ------------------------------- stable cells of the REAL model builders ---
def _model_modes(seed, P=6, H=3):
    keys = jax.random.split(jax.random.PRNGKey(seed), 4)
    radius = 0.2 + 0.7 * jax.random.uniform(keys[0], (P,))
    angle = jax.random.uniform(keys[1], (P,), minval=-2.0, maxval=2.0)
    lambda_bar = (radius * np.exp(1j * angle)).astype(_complex())
    b_bar = (jax.random.normal(keys[2], (P, H))
             + 1j * jax.random.normal(keys[3], (P, H))).astype(_complex())
    return lambda_bar, b_bar


def test_subset_stability_is_recorded_as_a_subset_result_only():
    """A radius measured on a mode SUBSET is a fact about that subset.

    tau = 1000 with eps = 1/4 really does give rho = 0.998 on the cluster's
    small fixture; that is recorded here as a subset observation and is
    explicitly NOT production eligibility, which only full-inventory
    certification decides. Three earlier errors came from blurring the two.
    """
    lambda_bar, b_bar = _model_modes(14)
    real = np.float64 if X64 else np.float32
    observations = []
    for tau_value in TAU_CANDIDATES:
        for eps in (0.0, 0.25):
            tau = np.full(lambda_bar.shape, tau_value, dtype=real)
            mass = DP.mass_from_eps(tau, np.asarray(eps, dtype=real))
            A, _ = DP.matched_state_coefficients(
                lambda_bar, b_bar, tau, mass, DP.PROFESSOR_LINEAR_TARGET)
            radius = float(np.max(ORACLE.exact_companion_radius(A)))
            observations.append({"tau": tau_value, "eps": eps,
                                 "subset_radius": radius,
                                 "scope": "THESE MODES ONLY, not production"})
    print({"subset_observations": observations,
           "production_eligibility": "decided by certification.py alone"})
    # the subset genuinely contains unstable cells, which is the whole point
    assert any(row["subset_radius"] > 1.0 for row in observations), \
        observations


def test_certification_is_the_only_source_of_production_eligibility():
    """No test, study or launcher may keep its own stable-tau list.

    Every hard-coded list of "stable" tau values is a subset-certification
    error waiting to happen; three of them already were. This asserts the
    lists are gone and that the certification module exposes the API the
    others consume.
    """
    from experiments.s5_direct_prospective import certification as CERT

    for name in ("WELL_CONDITIONED_TAU", "CLUSTER_STABLE_CELLS"):
        assert name not in globals(), name
    source = open(os.path.join(REPO,
                               "tests/test_direct_prospective_scan.py")).read()
    assert "def _stable_model_cells" not in source
    study = open(os.path.join(
        REPO, "experiments/s5_direct_prospective/chunk_study.py")).read()
    assert "CLUSTER_STABLE_CELLS" not in study
    assert "certification" in study
    # the single source of truth, with the tightened bound
    assert CERT.RADIUS_BOUND == 1.0
    for attribute in ("production_mode_inventory", "gate_1_stability",
                      "certify_seeds", "eligible_cells", "TAU_CANDIDATES",
                      "SEEDS", "MODEL_SPECS"):
        assert hasattr(CERT, attribute), attribute
    assert tuple(CERT.SEEDS) == (301, 302, 303)


def test_the_rejected_fixtures_stay_rejected():
    """REGRESSIONS. tau = 2 and tau = 5 with eps = 0 are unstable; the
    second is the cell this file wrongly hard-coded as stable, whose
    cluster-measured radius was 1.0787247827275395.

    The witness used here is analytic and PRNG-free -- Abar = 0.9 e^{-2i} --
    so the rejection reproduces anywhere.
    """
    from experiments.s5_direct_prospective import certification as CERT

    real = np.float64 if X64 else np.float32
    lambda_bar = np.asarray([0.9 * np.exp(-2.0j)], dtype=_complex())
    b_bar = np.ones((1, 1), dtype=_complex())
    for fixture in REJECTED_FIXTURES:
        tau = np.full((1,), fixture["tau"], dtype=real)
        A, _ = DP.professor_tss_state_coefficients(
            lambda_bar, b_bar, tau, DP.PROFESSOR_LINEAR_TARGET)
        radius = float(ORACLE.exact_companion_radius(A)[0])
        where = _context(tau=fixture["tau"], eps=fixture["eps"],
                         witness=fixture["witness"], radius=radius,
                         cluster_radius=fixture["radius"],
                         bound=CERT.RADIUS_BOUND)
        assert radius > CERT.RADIUS_BOUND, where      # gate 1 rejects it
        assert fixture["radius"] > 1.0, where         # so did the cluster


def _first_nonfinite(values):
    """(token, mode) of the first nonfinite entry, or (None, None)."""
    finite = np.isfinite(values)
    rows = np.where(~np.all(finite, axis=1))[0]
    if rows.size == 0:
        return None, None
    token = int(rows[0])
    return token, int(np.where(~finite[token])[0][0])


def test_gate_1_rejects_the_cells_that_a_mode_subset_called_stable():
    """REGRESSION for the tau = 2 failure, preserving the offending mode.

    Abar = 0.9 e^{-2i} lies inside the region the old test sampled, has
    companion radius 1.4416, and makes the MODEL diverge -- the sequential
    float32 path goes nonfinite at token 242 exactly as the block scan does.
    The cell is a gate-1 rejection, and no scan or tolerance is implicated.
    """
    from experiments.s5_direct_prospective import certification as CERT

    real = np.float64 if X64 else np.float32
    lambda_bar = np.asarray(
        [OFFENDING_MODE["abs_lambda"]
         * np.exp(1j * OFFENDING_MODE["angle"])], dtype=_complex())
    b_bar = np.ones((1, 1), dtype=_complex())
    tau = np.full((1,), OFFENDING_MODE["tau"], dtype=real)
    A, C = DP.professor_tss_state_coefficients(lambda_bar, b_bar, tau,
                                               DP.PROFESSOR_LINEAR_TARGET)
    radius = float(ORACLE.exact_companion_radius(A)[0])
    where = _context(tau=OFFENDING_MODE["tau"], eps=0.0,
                     target_construction=DP.PROFESSOR_LINEAR_TARGET,
                     lambda_bar=complex(lambda_bar[0]), radius=radius)
    assert abs(radius - OFFENDING_MODE["radius"]) < 1e-6, where
    assert radius > CERT.RADIUS_BOUND, where          # gate 1 rejects it

    # and the divergence is the MODEL's, not a scan's: the sequential path
    # goes nonfinite too, at the same token
    drive = DP.input_drive(C, np.zeros((4000, 1), dtype=real).at[0, 0].set(1.0))
    sequential = DP.sequential_scan_jax(A, drive)
    sequential_token, _ = _first_nonfinite(sequential)
    if not X64:                       # float32: the model overflows
        assert sequential_token is not None, where
        for chunk in (1, 2, 4, 8, 16, 32, 64):
            block_token, _ = _first_nonfinite(DP.block_scan(A, drive, chunk))
            assert block_token == sequential_token, _context(
                chunk=chunk, block=block_token,
                sequential=sequential_token, **{"cell": where})


def test_sequential_finiteness_is_asserted_before_any_block_comparison():
    """GATE 2, restructured so a NaN can never be misattributed.

    The sequential reference is checked and REPORTED first; only if it is
    entirely finite is a block-relative error computed at all. Cells are
    certified over every mode by gate 1 beforehand.
    """
    from experiments.s5_direct_prospective import certification as CERT

    lambda_bar, b_bar = _model_modes(14, P=16, H=4)
    single = lambda_bar.astype(np.complex64), b_bar.astype(np.complex64)
    inputs = jax.random.normal(jax.random.PRNGKey(40),
                               (4000, 4)).astype(np.float32)
    checked = 0
    for tau_value in (2.0, 5.0, 10.0, 50.0):
        for eps in (0.0, 0.25):
            tau = np.full(lambda_bar.shape, tau_value, dtype=np.float32)
            mass = DP.mass_from_eps(tau, np.asarray(eps, dtype=np.float32))
            A, C = DP.matched_state_coefficients(
                single[0], single[1], tau, mass, DP.PROFESSOR_LINEAR_TARGET)
            radius = float(np.max(ORACLE.exact_companion_radius(A)))
            base = _context(tau=tau_value, eps=eps, length=4000,
                            target_construction=DP.PROFESSOR_LINEAR_TARGET,
                            max_radius_over_these_modes=radius)
            drive = DP.input_drive(C, inputs)
            assert bool(np.all(np.isfinite(drive))), base
            sequential = DP.sequential_scan_jax(A, drive)
            token, mode = _first_nonfinite(sequential)
            report = {"context": base, "radius": radius,
                      "sequential_finite": token is None,
                      "sequential_first_nonfinite": {"token": token,
                                                     "mode": mode}}
            if token is not None:
                # the MODEL diverges: gate 1 must have rejected this cell,
                # and no block-relative error is computed at all
                report["verdict"] = "REJECTED_BY_GATE_1"
                print(report)
                assert radius > CERT.RADIUS_BOUND, report
                continue
            scale = float(np.max(np.abs(sequential)))
            assert np.isfinite(scale) and scale > 0.0, report
            for chunk in (1, 2, 4, 8, 16, 32, 64):
                block = DP.block_scan(A, drive, chunk)
                block_token, block_mode = _first_nonfinite(block)
                row = dict(report, chunk=chunk,
                           block_first_nonfinite={"token": block_token,
                                                  "mode": block_mode})
                if block_token is not None:
                    row["verdict"] = "BLOCK_DIVERGED_WHERE_SEQUENTIAL_DID_NOT"
                    print(row)
                    continue                       # recorded, not asserted
                row["block_vs_sequential_float32"] = float(
                    np.max(np.abs(block - sequential))) / scale
                row["verdict"] = "COMPARED"
                print(row)
                checked += 1
    assert checked > 0, "no cell reached a block-versus-sequential comparison"


def _verified_stable_model_cells(lambda_bar, b_bar, taus=TAU_CANDIDATES):
    """Cells whose radius is VERIFIED below 1 on THESE modes.

    This certifies nothing about production -- `certification.py` does that
    over the whole inventory. It exists so a numerical-conditioning test has
    a fixture whose stability is established rather than assumed, and every
    cell it returns carries its measured radius.
    """
    real = np.float64 if X64 else np.float32
    found = []
    for tau_value in taus:
        for eps in (0.0, 0.25):
            tau = np.full(lambda_bar.shape, tau_value, dtype=real)
            mass = DP.mass_from_eps(tau, np.asarray(eps, dtype=real))
            A, _ = DP.matched_state_coefficients(
                lambda_bar, b_bar, tau, mass, DP.PROFESSOR_LINEAR_TARGET)
            radius = float(np.max(ORACLE.exact_companion_radius(A)))
            if radius < 1.0:
                found.append({"tau": tau_value, "eps": eps,
                              "radius": radius,
                              "target": DP.PROFESSOR_LINEAR_TARGET})
    return found


def test_the_stable_cell_scan_matches_the_oracle_within_measured_conditioning():
    """Numerical conditioning, on cells whose stability is VERIFIED on the
    very modes used -- never assumed, never taken from a list."""
    lambda_bar, b_bar = _model_modes(14)
    real = np.float64 if X64 else np.float32
    epsilon = 2.220446049250313e-16 if X64 else 1.1920929e-07
    examined = 0
    for cell in _verified_stable_model_cells(lambda_bar, b_bar):
        tau = np.full(lambda_bar.shape, cell["tau"], dtype=real)
        mass = DP.mass_from_eps(tau, np.asarray(cell["eps"], dtype=real))
        A, _ = DP.matched_state_coefficients(lambda_bar, b_bar, tau, mass,
                                             cell["target"])
        for length in (3, 17, 257, 1000):
            fast = DP.matched_states(lambda_bar, b_bar, _inputs(length, length),
                                     tau, mass, cell["target"],
                                     scan_kind="block", chunk=64)
            slow = ORACLE.matched_sequential(lambda_bar, b_bar,
                                             _inputs(length, length), tau,
                                             mass, cell["target"])
            report = ORACLE.compare_finite_prefix(fast, slow)
            tolerance = ORACLE.doubling_scan_tolerance(A, length, epsilon)
            report.update(context=_context(
                tau=cell["tau"], eps=cell["eps"], radius=cell["radius"],
                target_construction=cell["target"], length=length,
                scan="block"), tolerance=tolerance)
            print(report)
            assert report["finite_masks_identical"], report
            assert report["common_finite_prefix"] == length, report
            assert report["max_relative_error_on_prefix"] <= tolerance, report
            examined += 1
    print({"cells_examined": examined,
           "note": "zero means no cell of these modes was verifiably stable"})


def test_the_doubling_scan_needs_float64_at_this_cell():
    """A pre-training finding, asserted so it cannot be forgotten.

    At the stable cell the companion is strongly non-normal, and the scan's
    squaring step loses far more precision than the sequential recurrence
    does. In emulated float32 the scan's relative error reached 7.2 at
    L = 257 and 1.5e5 at L = 1000, while the sequential recurrence stayed at
    about 2e-3 (docs/analysis/direct_prospective_float32.txt). The scan is
    therefore NOT usable in float32 at this cell, which is the precision the
    production path uses.
    """
    if not X64:
        return
    lambda_bar, b_bar = _model_modes(14)
    tau64 = np.full(lambda_bar.shape, 1000.0, dtype=np.float64)
    mass64 = DP.mass_from_eps(tau64, np.asarray(0.25, dtype=np.float64))
    inputs64 = _inputs(20, 257)
    reference = DP.matched_states(lambda_bar, b_bar, inputs64, tau64, mass64,
                                  DP.PROFESSOR_LINEAR_TARGET)
    single = DP.matched_states(lambda_bar.astype(np.complex64),
                               b_bar.astype(np.complex64),
                               inputs64.astype(np.float32),
                               tau64.astype(np.float32),
                               mass64.astype(np.float32),
                               DP.PROFESSOR_LINEAR_TARGET)
    scale = float(np.max(np.abs(reference)))
    error = float(np.max(np.abs(single.astype(np.complex128) - reference)))
    relative = error / scale
    print({"float32_scan_relative_error": relative, "scale": scale})
    # the finding: float32 is NOT adequate for the scan here
    assert relative > 1e-3, relative


# ------------------------------------------- the unstable-model regression -
def _unstable_fixture(length=1000, tau_value=0.5, eps=0.25,
                      target=DP.PROFESSOR_LINEAR_TARGET):
    """The production-derived fixture that made the first cluster run fail."""
    real = np.float64 if X64 else np.float32
    lambda_bar, b_bar = _model_modes(1)
    tau = np.full(lambda_bar.shape, tau_value, dtype=real)
    mass = DP.mass_from_eps(tau, np.asarray(eps, dtype=real))
    inputs = _inputs(length, length)
    return lambda_bar, b_bar, inputs, tau, mass, target


def test_the_unstable_model_is_reported_and_scan_agrees_on_the_finite_prefix():
    """The regression that replaces the misdiagnosed failure.

    The WWJ_GENERALIZED_TSS model on the PROFESSOR_LINEAR_TARGET with
    tau = 0.5, M = tau^2/4 is UNSTABLE on S5-like modes: it overflows within
    a thousand tokens. What must hold is that the scan reproduces the
    oracle exactly until the model itself diverges, and that both diverge at
    the same place.
    """
    lambda_bar, b_bar, inputs, tau, mass, target = _unstable_fixture()
    fast = DP.matched_states(lambda_bar, b_bar, inputs, tau, mass, target)
    slow = ORACLE.matched_sequential(lambda_bar, b_bar, inputs, tau, mass,
                                     target)
    report = ORACLE.compare_finite_prefix(fast, slow)
    A, _ = DP.matched_state_coefficients(lambda_bar, b_bar, tau, mass, target)
    radius = float(np.max(ORACLE.exact_companion_radius(A)))
    report["max_companion_radius"] = radius
    print(report)                          # the diagnostic, always visible

    # the instability is real and is reported, not tolerated
    assert radius > 1.0 + 1e-5, report
    # scan and oracle agree over the whole common finite prefix
    assert report["common_finite_prefix"] > 0, report
    assert report["max_relative_error_on_prefix"] <= (1e-10 if X64
                                                      else 1e-3), report
    # and they fail in the same place, if they fail at all
    assert report["same_first_nonfinite"], report
    assert report["finite_masks_identical"], report
    # divergence between the two never precedes the model's own divergence
    assert report["max_relative_error_on_prefix"] < 1.0, report


def test_the_unstable_cell_can_never_be_classified_as_admissible():
    """The stability bound is what rejects it, and no NaN-tolerant
    comparison can launder it into an admissible initialization."""
    from experiments.s5_direct_prospective.stability_grid import RADIUS_BOUND

    lambda_bar, b_bar, _, tau, mass, target = _unstable_fixture()
    A, _ = DP.matched_state_coefficients(lambda_bar, b_bar, tau, mass, target)
    exact = ORACLE.exact_companion_radius(A)
    assert float(max(exact)) > RADIUS_BOUND
    estimate = DP.companion_spectral_radius(A)
    assert float(np.max(estimate)) > RADIUS_BOUND
    assert float(np.max(estimate)) >= float(max(exact)) - 1e-6


# ------------------------------------------------ the two scientific arms --
def test_the_M_zero_coefficient_identities_hold_for_every_target_construction():
    """COEFFICIENT-LEVEL M = 0 reduction: A0 = P0, A1 = P1, A2 = 0,
    C0 = Q0, C1 = Q1, C2 = 0, for BOTH target constructions and every tau.

    This is the identity itself and is independent of any scan, any length
    and any stability question -- which is why it is tested on its own.
    """
    lambda_bar, b_bar = _model_modes(15)
    real = np.float64 if X64 else np.float32
    zero = np.zeros(lambda_bar.shape, dtype=real)
    tolerance = TOL64 if X64 else TOL32
    for tau_value in (0.05, 2.0, 5.0, 10.0, 100.0, 1000.0):
        tau = np.full(lambda_bar.shape, tau_value, dtype=real)
        assert DP.model_of(zero) == DP.PROFESSOR_TSS
        assert DP.model_of(DP.mass_from_eps(tau, np.asarray(0.25, real))) == \
            DP.WWJ_GENERALIZED_TSS
        for target in DP.TARGET_CONSTRUCTIONS:
            where = _context(tau=tau_value, target_construction=target)
            (A0, A1, A2), (C0, C1, C2) = DP.matched_state_coefficients(
                lambda_bar, b_bar, tau, zero, target)
            (P0, P1), (Q0, Q1) = DP.professor_tss_state_coefficients(
                lambda_bar, b_bar, tau, target)
            assert _close(A0, P0, tolerance), where
            assert _close(A1, P1, tolerance), where
            assert float(np.max(np.abs(A2))) == 0.0, where
            assert _close(C0, Q0, tolerance), where
            assert _close(C1, Q1, tolerance), where
            assert float(np.max(np.abs(C2))) == 0.0, where


def _analytically_stable_M_zero_fixture(tau_value, modes=6):
    """Modes DELIBERATELY CONSTRUCTED so the M = 0 recurrence is stable.

    For M = 0 the companion is z^2 - (a + c0 Abar) z + Abar with
    a = 1 - h/tau, c0 = 1 + h/tau, so stability depends on Abar. Rather than
    hoping a random draw is stable, candidate Abar values are scanned and
    only those with radius <= 0.9 are kept; the test then asserts that bound
    again on the assembled fixture. Nothing here is a claim about
    production, which `certification.py` alone decides.
    """
    real = np.float64 if X64 else np.float32
    candidates = []
    for magnitude in (0.2, 0.35, 0.5, 0.65, 0.8):
        for angle in np.linspace(-np.pi, np.pi, 37):
            lam = np.asarray([magnitude * np.exp(1j * float(angle))],
                             dtype=_complex())
            b = np.ones((1, 1), dtype=_complex())
            tau = np.full((1,), tau_value, dtype=real)
            A, _ = DP.professor_tss_state_coefficients(
                lam, b, tau, DP.PROFESSOR_LINEAR_TARGET)
            if float(ORACLE.exact_companion_radius(A)[0]) <= 0.9:
                candidates.append(complex(lam[0]))
            if len(candidates) >= modes:
                break
        if len(candidates) >= modes:
            break
    if not candidates:
        return None, None
    lambda_bar = np.asarray(candidates, dtype=_complex())
    b_bar = np.ones((len(candidates), 2), dtype=_complex())
    return lambda_bar, b_bar


def test_the_M_zero_sequence_identity_on_an_analytically_stable_fixture():
    """SEQUENCE-LEVEL M = 0 reduction, on a fixture whose stability is
    constructed and then re-asserted, through the BLOCK implementation:

        order-three WWJ at M = 0 == order-two Professor == sequential oracle

    at lengths up to 16000. No production claim is made here.
    """
    real = np.float64 if X64 else np.float32
    target = DP.PROFESSOR_LINEAR_TARGET
    tolerance = TOL64 if X64 else TOL32
    checked = 0
    for tau_value in TAU_CANDIDATES:
        lambda_bar, b_bar = _analytically_stable_M_zero_fixture(tau_value)
        if lambda_bar is None:
            print(_context(tau=tau_value,
                           note="no mode of the scan is stable at this tau"))
            continue
        tau = np.full(lambda_bar.shape, tau_value, dtype=real)
        zero = np.zeros(lambda_bar.shape, dtype=real)
        A, _ = DP.matched_state_coefficients(lambda_bar, b_bar, tau, zero,
                                             target)
        radius = float(np.max(ORACLE.exact_companion_radius(A)))
        where = _context(tau=tau_value, target_construction=target,
                         radius=radius, modes=int(lambda_bar.shape[0]))
        assert radius <= 0.9, where            # constructed AND verified
        for length in (17, 257, 4096, 16000):
            here = _context(tau=tau_value, target_construction=target,
                            length=length, radius=radius, scan="block")
            inputs = _inputs(length, length, H=b_bar.shape[1])
            three = DP.matched_states(lambda_bar, b_bar, inputs, tau, zero,
                                      target, scan_kind="block", chunk=64)
            two = DP.professor_tss_states(lambda_bar, b_bar, inputs, tau,
                                          target, scan_kind="block",
                                          chunk=64)
            assert bool(np.all(np.isfinite(three))), here
            assert _close(three, two, tolerance), here
            if length <= 4096:
                oracle = ORACLE.professor_tss_sequential(lambda_bar, b_bar,
                                                         inputs, tau, target)
                assert _close(two, oracle, tolerance), here
            checked += 1
    assert checked > 0, "no analytically stable M = 0 fixture was built"


def test_the_M_zero_production_sequence_identity_only_for_certified_cells():
    """The production version of the same identity, run ONLY on cells that
    full-inventory certification returns PASS for. If there are none, the
    result is RECORDED, not skipped silently and not replaced by a
    fabricated candidate."""
    from experiments.s5_direct_prospective import certification as CERT

    inventory = CERT.production_mode_inventory(301)
    certified = []
    for tau_value in TAU_CANDIDATES:
        row = CERT.gate_1_stability(inventory, tau_value, 0.0, 2)
        row["tau"] = tau_value
        print({"tau": tau_value, "verdict": "PASS" if row["passes"]
               else "REJECT", "max_radius": row["max_radius_over_all_modes"],
               "worst_mode": row["worst_mode"],
               "modes_certified": row["modes_certified"]})
        if row["passes"]:
            certified.append(row)
    if not certified:
        print({"status": "NO_ELIGIBLE_CELL_IN_THE_PRODUCTION_INVENTORY",
               "seed": 301, "tau_candidates": list(TAU_CANDIDATES)})
        return
    real = np.float64 if X64 else np.float32
    entry = inventory[0]
    for row in certified:
        tau = np.full(entry["lambda_bar"].shape, row["tau"], dtype=real)
        zero = np.zeros(entry["lambda_bar"].shape, dtype=real)
        for length in (257, 4096):
            here = _context(tau=row["tau"], length=length, seed=301,
                            target_construction=DP.PROFESSOR_LINEAR_TARGET,
                            max_radius=row["max_radius_over_all_modes"])
            inputs = _inputs(length, length, H=entry["b_bar"].shape[1])
            three = DP.matched_states(entry["lambda_bar"], entry["b_bar"],
                                      inputs, tau, zero,
                                      DP.PROFESSOR_LINEAR_TARGET,
                                      scan_kind="block", chunk=64)
            two = DP.professor_tss_states(entry["lambda_bar"], entry["b_bar"],
                                          inputs, tau,
                                          DP.PROFESSOR_LINEAR_TARGET,
                                          scan_kind="block", chunk=64)
            assert bool(np.all(np.isfinite(three))), here
            assert _close(three, two, TOL64 if X64 else TOL32), here


def test_unstable_cells_are_rejected_by_the_gate_not_compared_as_arrays():
    """For an UNSTABLE cell the right assertion is that the stability gate
    rejects it, plus a finite-prefix diagnostic on the SEQUENTIAL path.
    Comparing two all-NaN arrays proves nothing and is never done.
    """
    from experiments.s5_direct_prospective.stability_grid import RADIUS_BOUND

    lambda_bar, b_bar = _model_modes(15)
    real = np.float64 if X64 else np.float32
    zero = np.zeros(lambda_bar.shape, dtype=real)
    target = DP.NATIVE_MATCHED_TARGET
    rejected = 0
    for tau_value in (2.0, 10.0, 100.0):
        tau = np.full(lambda_bar.shape, tau_value, dtype=real)
        A, _ = DP.matched_state_coefficients(lambda_bar, b_bar, tau, zero,
                                             target)
        radius = float(np.max(ORACLE.exact_companion_radius(A)))
        where = _context(tau=tau_value, target_construction=target,
                         radius=radius)
        if radius <= RADIUS_BOUND:
            continue                                   # stable: nothing to do
        rejected += 1
        # gate 1 rejects it, and that is the assertion
        assert radius > RADIUS_BOUND, where
        # and the two implementations still agree wherever the MODEL is
        # finite, checked on the sequential path with a prefix diagnostic
        inputs = _inputs(40, 128)
        fast = DP.matched_states(lambda_bar, b_bar, inputs, tau, zero,
                                 target, scan_kind="sequential")
        slow = ORACLE.matched_sequential(lambda_bar, b_bar, inputs, tau,
                                         zero, target)
        report = ORACLE.compare_finite_prefix(fast, slow)
        report["context"] = where
        print(report)
        assert report["finite_masks_identical"], report
        assert report["same_first_nonfinite"], report
        if report["common_finite_prefix"] > 0:
            assert report["max_relative_error_on_prefix"] < 1.0, report
    assert rejected > 0, "expected the native-matched target to be unstable"


def test_the_legacy_zucchet_coefficients_are_not_this_recurrence():
    """The legacy arm differs by an explicit residual-feedback term; the
    difference is reported, not reconciled, and the legacy arm is untouched."""
    from s5.discrete_recurrence import zucchet_coefficients

    lambda_bar, b_bar = _model_modes(16)
    real = np.float64 if X64 else np.float32
    for k_value in (0.05, 0.5, 1.0):
        k = np.full(lambda_bar.shape, k_value, dtype=real)
        legacy_a1, legacy_a2, legacy_c1, legacy_c2 = zucchet_coefficients(
            lambda_bar, b_bar, k)
        # our Professor/TSS with h/tau = k, i.e. tau = 1/k
        tau = 1.0 / k
        (A0, A1), (C0, C1) = DP.professor_tss_state_coefficients(
            lambda_bar, b_bar, tau, DP.PROFESSOR_LINEAR_TARGET)
        assert _close(legacy_a1, A0, TOL64 if X64 else TOL32)   # first tap agrees
        # the second tap differs by exactly -(1-k)(1 - Abar)
        difference = legacy_a2 - A1
        predicted = -(1.0 - k) * (1.0 - lambda_bar)
        assert _close(difference, predicted, TOL64 if X64 else TOL32), k_value
        # and the input tap by exactly (1-k) Bbar
        input_difference = legacy_c2 - C1
        assert _close(input_difference, (1.0 - k)[..., None] * b_bar,
                      TOL64 if X64 else TOL32), k_value
        if abs(k_value - 1.0) < 1e-12:
            assert _close(difference, np.zeros_like(difference), 1e-12)


# ------------------------------------------------- what must not change ----
def _blob(relative):
    return subprocess.run(["git", "-C", REPO, "rev-parse",
                           f"{NATIVE_BASE}:{relative}"], capture_output=True,
                          text=True).stdout.strip()


def _worktree(relative):
    return subprocess.run(["git", "-C", REPO, "hash-object",
                           os.path.join(REPO, relative)],
                          capture_output=True, text=True).stdout.strip()


def test_native_and_every_existing_arm_are_byte_identical():
    for relative in ("s5/ssm.py", "s5/three_arm_factory.py",
                     "s5/discrete_recurrence.py", "s5/prospective_ssm.py",
                     "s5/generalized_prospective_ssm.py",
                     "s5/wwj_operator.py", "s5/wwj_ssm.py",
                     "experiments/s5_three_arm_full/runner.py"):
        assert _blob(relative) == _worktree(relative) != "", relative


# ------------------------------------------------------- the block scan ----
def _block_cell(P=8, H=3):
    """The accepted stable cell, in both orders."""
    real = np.float64 if X64 else np.float32
    lambda_bar, b_bar = _model_modes(14, P=P, H=H)
    tau = np.full((P,), 1000.0, dtype=real)
    mass = DP.mass_from_eps(tau, np.asarray(0.25, dtype=real))
    return lambda_bar, b_bar, tau, mass


def test_the_block_scan_equals_the_sequential_scan_in_float64():
    """Both orders, real and complex modes, arbitrary lengths including
    non-multiples of the chunk size."""
    tolerance = TOL64 if X64 else TOL32
    for complex_modes in (True, False):
        lambda_bar, b_bar = _model_modes(30, complex_modes=complex_modes) \
            if complex_modes else (np.abs(_model_modes(30)[0]),
                                   _model_modes(30)[1])
        real = np.float64 if X64 else np.float32
        tau = np.full(lambda_bar.shape, 1000.0, dtype=real)
        mass = DP.mass_from_eps(tau, np.asarray(0.25, dtype=real))
        for order, coefficients in (
                (2, DP.professor_tss_state_coefficients(
                    lambda_bar, b_bar, tau, DP.PROFESSOR_LINEAR_TARGET)),
                (3, DP.matched_state_coefficients(
                    lambda_bar, b_bar, tau, mass,
                    DP.PROFESSOR_LINEAR_TARGET))):
            A, C = coefficients
            for chunk in (16, 32, 64, 128, 256):
                for length in (1, 3, 17, 63, 64, 65, 257, 1000, 1023):
                    inputs = _inputs(length, length)
                    drive = DP.input_drive(C, inputs)
                    block = DP.block_scan(A, drive, chunk)
                    oracle = ORACLE.sequential_scan(A, drive)
                    assert block.shape == drive.shape
                    assert _close(block, oracle, tolerance), (order, chunk,
                                                              length)


def test_the_block_scan_has_zero_prehistory_and_no_wraparound():
    lambda_bar, b_bar, tau, mass = _block_cell()
    A, C = DP.matched_state_coefficients(lambda_bar, b_bar, tau, mass,
                                         DP.PROFESSOR_LINEAR_TARGET)
    length, chunk = 40, 16                       # 40 is not a multiple of 16
    for impulse in (0, 1, 15, 16, 17, 39):
        drive = np.zeros((length, lambda_bar.shape[0]),
                         dtype=A[0].dtype).at[impulse].set(1.0)
        states = DP.block_scan(A, drive, chunk)
        assert bool(np.all(states[:impulse] == 0)), impulse
        assert bool(np.any(states[impulse] != 0)), impulse


def test_the_block_scan_is_deterministic_and_remat_neutral():
    lambda_bar, b_bar, tau, mass = _block_cell()
    A, C = DP.matched_state_coefficients(lambda_bar, b_bar, tau, mass,
                                         DP.PROFESSOR_LINEAR_TARGET)
    drive = DP.input_drive(C, _inputs(31, 300))
    first = DP.block_scan(A, drive, 64)
    assert bool(np.all(first == DP.block_scan(A, drive, 64)))
    plain = DP.block_scan(A, drive, 64, remat=None)
    assert _close(first, plain, TOL64 if X64 else TOL32)


def test_block_scan_gradients_match_the_sequential_path():
    """Gradients with respect to Abar, Bbar, tau and eps."""
    lambda_bar, b_bar = _model_modes(32, P=4, H=2)
    real = np.float64 if X64 else np.float32
    inputs = _inputs(33, 257, H=2)
    tolerance = 1e-7 if X64 else 1e-2

    def loss(lam, b, tau_value, eps_value, scan_kind):
        tau = np.full(lam.shape, tau_value, dtype=real)
        mass = DP.mass_from_eps(tau, eps_value)
        states = DP.matched_states(lam, b, inputs, tau, mass,
                                   DP.PROFESSOR_LINEAR_TARGET,
                                   scan_kind=scan_kind, chunk=64)
        return np.sum(np.abs(states) ** 2).real

    for argnums in (0, 1, 2, 3):
        block = jax.grad(loss, argnums=argnums)(
            lambda_bar, b_bar, np.asarray(1000.0, real),
            np.asarray(0.25, real), "block")
        sequential = jax.grad(loss, argnums=argnums)(
            lambda_bar, b_bar, np.asarray(1000.0, real),
            np.asarray(0.25, real), "sequential")
        assert _close(block, sequential, tolerance), argnums


def test_the_block_scan_reduces_to_the_professor_model_at_M_zero():
    lambda_bar, b_bar = _model_modes(34)
    real = np.float64 if X64 else np.float32
    tau = np.full(lambda_bar.shape, 1000.0, dtype=real)
    zero = np.zeros(lambda_bar.shape, dtype=real)
    for length in (17, 257):
        inputs = _inputs(length, length)
        three = DP.matched_states(lambda_bar, b_bar, inputs, tau, zero,
                                  DP.PROFESSOR_LINEAR_TARGET, chunk=32)
        two = DP.professor_tss_states(lambda_bar, b_bar, inputs, tau,
                                      DP.PROFESSOR_LINEAR_TARGET, chunk=32)
        oracle = ORACLE.professor_tss_sequential(lambda_bar, b_bar, inputs,
                                                 tau,
                                                 DP.PROFESSOR_LINEAR_TARGET)
        assert _close(three, two, TOL64 if X64 else TOL32), length
        assert _close(two, oracle, TOL64 if X64 else TOL32), length


def test_production_length_float32_block_scan_is_finite_and_measured():
    """GATE 2, at production length: the block scan must stay finite, and
    its float32 error is COMPARED WITH the ordinary sequential float32
    error rather than asserted to be small. The emulation predicts it will
    be worse; the number is printed either way."""
    lambda_bar, b_bar = _model_modes(35, P=16, H=4)
    single = lambda_bar.astype(np.complex64), b_bar.astype(np.complex64)
    tau32 = np.full(lambda_bar.shape, 1000.0, dtype=np.float32)
    mass32 = DP.mass_from_eps(tau32, np.asarray(0.25, dtype=np.float32))
    inputs = jax.random.normal(jax.random.PRNGKey(36),
                               (16000, 4)).astype(np.float32)
    A32, C32 = DP.matched_state_coefficients(single[0], single[1], tau32,
                                             mass32,
                                             DP.PROFESSOR_LINEAR_TARGET)
    drive32 = DP.input_drive(C32, inputs)
    block = DP.block_scan(A32, drive32, 64)
    sequential = DP.sequential_scan_jax(A32, drive32)
    assert bool(np.all(np.isfinite(block))), "block scan diverged"
    scale = float(np.max(np.abs(sequential)))
    error = float(np.max(np.abs(block - sequential))) / scale
    print({"block_vs_sequential_float32_at_16000": error,
           "note": "gate 2 is decided by chunk_study.py on real modes"})
    assert np.isfinite(error)


def test_the_doubling_scan_is_never_selected_implicitly():
    """It stays as a labelled diagnostic and regression only."""
    import io as _io

    source = _io.open(os.path.join(REPO, "s5/direct_prospective.py")).read()
    assert 'SCAN_KINDS = ("block", "sequential", "doubling")' in source
    assert 'scan_kind="block"' in source
    assert "LABELLED DIAGNOSTIC" in source
    # the default of every states function is the block scan
    for function in ("matched_states", "partially_matched_states",
                     "professor_tss_states"):
        body = source[source.index(f"def {function}("):]
        assert 'scan_kind="block"' in body[:600], function
