"""The factorization, checked as algebra. No JAX, runs on a laptop.

The equation is not being changed, so what needs proving is that the two
parallel routes compute the SAME recurrence the sequential oracle does,
including where a naive factorization would fail: repeated roots,
near-repeated roots, and the cancellation that destroys the small root.
"""

import ast
import cmath
import io
import math
import os
import random

from tests import two_compartment_reference as R
from tests import source_introspection as SI

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACTORED = os.path.join(REPO, "s5/factored_recurrence.py")
DISCRETE = os.path.join(REPO, "s5/discrete_recurrence.py")
GENERALIZED = os.path.join(REPO, "s5/generalized_prospective_ssm.py")


def _pairs(count, seed=5, scale=1.0):
    """Random (a1, a2) with complex entries, the production shape."""
    generator = random.Random(seed)

    def value():
        return complex(generator.uniform(-scale, scale),
                       generator.uniform(-scale, scale))

    return [(value(), value()) for _ in range(count)]


# ------------------------------------------------- coefficient algebra --
def test_the_roots_reproduce_the_coefficients():
    """r1 + r2 = a1 and r1 r2 = -a2, which is the whole factorization."""
    for a1, a2 in _pairs(500):
        first, second = R.companion_roots(a1, a2)
        assert abs(first + second - a1) < 1e-10 * max(1.0, abs(a1))
        assert abs(first * second + a2) < 1e-10 * max(1.0, abs(a2))


def test_the_transfer_functions_are_identical():
    """1/(1 - a1 z^-1 - a2 z^-2) == 1/((1-r1 z^-1)(1-r2 z^-1))."""
    for a1, a2 in _pairs(200, seed=6, scale=0.6):
        for angle in (0.1, 0.7, 1.9, 3.0):
            z = 1.3 * cmath.exp(1j * angle)
            direct = R.transfer(a1, a2, z)
            split = R.factored_transfer(a1, a2, z)
            assert abs(direct - split) < 1e-9 * max(1.0, abs(direct))


def test_the_radius_agrees_with_the_existing_companion_radius():
    """`companion_radius` is production code and is not being changed, so
    the factored radius must agree with it rather than replace it."""
    for a1, a2 in _pairs(500, seed=7):
        first, second = R.companion_roots(a1, a2)
        assert abs(max(abs(first), abs(second))
                   - R.companion_radius(a1, a2)) < 1e-9


# --------------------------------------------------- numerical stability --
def test_the_stable_form_beats_the_naive_one_under_cancellation():
    """The reason the product form is used. With |a1| huge and |a2| tiny
    the small root is a1-sqrt(...) over 2, a difference of two nearly equal
    numbers, and the naive formula loses it."""
    worst_naive, worst_stable = 0.0, 0.0
    for scale in (1e6, 1e8, 1e10):
        a1, a2 = complex(scale), complex(1e-3)
        exact_small = -a2 / complex(scale)       # to leading order
        naive = min(R.naive_roots(a1, a2), key=abs)
        stable = min(R.companion_roots(a1, a2), key=abs)
        worst_naive = max(worst_naive,
                          abs(naive - exact_small) / abs(exact_small))
        worst_stable = max(worst_stable,
                           abs(stable - exact_small) / abs(exact_small))
    assert worst_stable < 1e-9, worst_stable
    assert worst_naive > 1e-3, worst_naive


def test_the_major_root_is_always_the_larger_one():
    """Deterministic ordering: the first returned root is never smaller."""
    for a1, a2 in _pairs(1000, seed=8):
        first, second = R.companion_roots(a1, a2)
        assert abs(first) >= abs(second) - 1e-12


def test_both_roots_vanish_only_when_both_coefficients_do():
    assert R.companion_roots(0j, 0j) == (0j, 0j)
    first, second = R.companion_roots(0j, complex(0.25))
    assert abs(abs(first) - 0.5) < 1e-12 and abs(abs(second) - 0.5) < 1e-12


# ------------------------------------------------------- repeated roots --
def test_repeated_roots_are_exact_rather_than_special_cased():
    """a2 = -a1^2/4 gives a double root at a1/2. The cascade is then the
    Jordan realization and reproduces t r^{t-1} exactly; a partial-fraction
    route would divide by (r1 - r2) and fail here."""
    for value in (0.3, -0.7, 0.95):
        a1 = complex(2.0 * value)
        a2 = -a1 * a1 / 4.0
        first, second = R.companion_roots(a1, a2)
        assert abs(first - value) < 1e-12 and abs(second - value) < 1e-12
        drive = [1.0 + 0j] + [0j] * 39
        direct = R.sequential(a1, a2, drive)
        split = R.factored(a1, a2, drive)
        for index, (want, got) in enumerate(zip(direct, split)):
            assert abs(want - got) < 1e-9 * max(1.0, abs(want))
            # the analytic impulse response of a double root
            expected = (index + 1) * value ** index
            assert abs(want - expected) < 1e-8 * max(1.0, abs(expected))


def test_near_repeated_roots_do_not_degrade():
    """The dangerous neighbourhood: the discriminant is tiny but nonzero."""
    for epsilon in (1e-6, 1e-9, 1e-12, 0.0):
        a1 = complex(1.2)
        a2 = -a1 * a1 / 4.0 + epsilon
        drive = [complex(math.sin(k)) for k in range(64)]
        direct = R.sequential(a1, a2, drive)
        split = R.factored(a1, a2, drive)
        scale = max(abs(v) for v in direct) or 1.0
        worst = max(abs(w - g) for w, g in zip(direct, split))
        assert worst < 1e-8 * scale, (epsilon, worst / scale)


# ------------------------------------------------- sequence equivalence --
def test_the_cascade_reproduces_the_sequential_recurrence():
    for a1, a2 in _pairs(120, seed=11, scale=0.7):
        drive = [complex(math.cos(k / 3.0), math.sin(k / 5.0))
                 for k in range(200)]
        direct = R.sequential(a1, a2, drive)
        split = R.factored(a1, a2, drive)
        scale = max(abs(v) for v in direct) or 1.0
        for want, got in zip(direct, split):
            assert abs(want - got) < 1e-8 * scale


def test_the_cascade_does_not_care_which_root_runs_first():
    """Order-independence, which is why a deterministic ordering is enough
    and a canonical one is not required."""
    for a1, a2 in _pairs(60, seed=13, scale=0.7):
        first, second = R.companion_roots(a1, a2)
        drive = [complex(math.sin(k / 2.0)) for k in range(120)]

        def scan(pole, values):
            out, carry = [], 0j
            for value in values:
                carry = pole * carry + value
                out.append(carry)
            return out

        forward = scan(second, scan(first, drive))
        swapped = scan(first, scan(second, drive))
        scale = max(abs(v) for v in forward) or 1.0
        for want, got in zip(forward, swapped):
            assert abs(want - got) < 1e-8 * scale


def test_zero_prehistory_and_no_shifted_input_wraparound():
    """The drive is c1 x_t + c2 x_{t-1} with NOTHING before t = 0, so an
    impulse at t cannot influence any earlier state."""
    inputs = [0j] * 10 + [1.0 + 0j] + [0j] * 10
    drive = R.drive_from_inputs(0.7 + 0.1j, -0.3 + 0.2j, inputs)
    assert all(value == 0j for value in drive[:10])
    states = R.sequential(0.5 + 0.2j, -0.1 + 0.05j, drive)
    assert all(value == 0j for value in states[:10])
    assert abs(states[10]) > 0.0


# --------------------------------------------------------- the module --
def test_the_equation_and_its_parameterization_are_untouched():
    """The brief: do not change `generalized_coefficients`, `target_map`,
    `response_mass_gamma`, or Native S5."""
    tree = SI.parse(DISCRETE)
    for name in ("target_map", "zucchet_coefficients",
                 "generalized_coefficients", "scan_companion",
                 "scan_companion_sequential", "companion_radius"):
        assert SI.defines_function(tree, name), name
    assert SI.defines_function(SI.parse(GENERALIZED), "response_mass_gamma")


def test_the_sequential_scan_remains_the_default():
    source = io.open(FACTORED).read()
    tree = SI.parse(FACTORED)
    assert 'DEFAULT_IMPLEMENTATION = "sequential"' in source
    names = dict(SI.imported_names(tree))
    assert "scan_companion_sequential" in {n for _, n in
                                           SI.imported_names(tree)}
    for name in ("companion_roots", "scan_factored", "scan_for",
                 "spectral_radius_from_roots"):
        assert SI.defines_function(tree, name), name


def test_no_eigendecomposition_in_the_training_path():
    """Explicitly required. Roots come from the quadratic formula, not from
    a linear-algebra routine."""
    source = io.open(FACTORED).read()
    for forbidden in ("eig", "eigvals", "eigh", "svd", "linalg"):
        assert forbidden not in source, forbidden


def test_no_file_reads_a_name_nothing_binds():
    for path in (FACTORED, DISCRETE, GENERALIZED):
        assert SI.undefined_names(path) == {}, (path,
                                                SI.undefined_names(path))


# ---------------------------------------------------------- byte identity --
#: the ONE pre-existing file this branch is allowed to touch, and why:
#: the arm needs an explicit implementation choice, and the default keeps
#: the production path on the sequential oracle.
JUSTIFIED_CHANGES = {"s5/generalized_prospective_ssm.py"}
BASE = "ef004cda025b4e098041cfe5970c217fa0015bba"


def _changed_tracked_files():
    """Files that existed at the base commit AND have since changed.

    Intersected with the base tree deliberately: a plain `git diff --name-only`
    also lists files this branch ADDED, which are not identity violations.
    """
    import subprocess

    def run(*arguments):
        finished = subprocess.run(("git",) + arguments, cwd=REPO,
                                  capture_output=True, text=True)
        assert finished.returncode == 0, finished.stderr
        return set(finished.stdout.split())

    tracked_at_base = run("ls-tree", "-r", "--name-only", BASE)
    changed = run("diff", "--name-only", "--diff-filter=MDRTC", BASE, "HEAD")
    return tracked_at_base & changed


def test_native_and_every_other_production_file_are_byte_identical():
    """Native S5, the coefficients, the target map and the runner must be
    untouched. Only the one justified file may differ."""
    changed = _changed_tracked_files()
    assert changed <= JUSTIFIED_CHANGES, sorted(changed - JUSTIFIED_CHANGES)
    for path in ("s5/ssm.py", "s5/discrete_recurrence.py",
                 "experiments/s5_three_arm_full/runner.py",
                 "s5/three_arm_factory.py", "s5/prospective_ssm.py"):
        assert path not in changed, path


def test_the_one_justified_change_only_adds_an_implementation_choice():
    """It must not alter the equation, the coefficients or the default."""
    import subprocess
    finished = subprocess.run(
        ("git", "diff", "-U0", BASE, "HEAD", "--",
         "s5/generalized_prospective_ssm.py"),
        cwd=REPO, capture_output=True, text=True)
    assert finished.returncode == 0, finished.stderr
    removed = [line[1:].strip() for line in finished.stdout.splitlines()
               if line.startswith("-") and not line.startswith("---")]
    # nothing that defines the equation or its parameterization is removed
    for forbidden in ("def response_mass_gamma", "generalized_coefficients(",
                      "RHO_MIN", "response_init", "rho_init", "gamma_init"):
        assert not any(forbidden in line and "=" in line and "implementation"
                       not in line for line in removed
                       if line.startswith("def ") or "RHO_MIN" in line), (
            forbidden, removed)
    source = io.open(os.path.join(REPO,
                                  "s5/generalized_prospective_ssm.py")).read()
    assert "implementation: str = DEFAULT_IMPLEMENTATION" in source
    assert "scan_for(self.implementation)" in source
    # the parameterization constants are still exactly what they were
    for expected in ("response_init: float = 0.05", "rho_init: float = 0.5",
                     "gamma_init: float = 1.0", "RHO_MIN = 1e-4"):
        assert expected in source, expected


# ------------------------------------------------------------- benchmark --
BENCHMARK = os.path.join(REPO, "experiments/s5_two_compartment/benchmark.py")


def test_the_benchmark_does_not_set_the_scan_by_class_attribute():
    """REGRESSION, and it silently voided three rows of the first run.

    A Flax nn.Module is turned into a dataclass at CLASS DEFINITION time,
    so `__init__` has already captured the field defaults and assigning to
    the class attribute afterwards changes nothing an instance sees. Every
    generalized row ran the sequential scan; the giveaway was
    `peak_device_bytes` identical to the BYTE between `sequential` and
    `companion`, 8957147904 both, which two different algorithms cannot
    produce.
    """
    source = io.open(BENCHMARK).read()
    tree = SI.parse(BENCHMARK)
    # by AST, not by substring: the literal now appears in the docstring
    # that EXPLAINS the bug, so a substring search can never pass. That is
    # the exact trap `source_introspection` was written for.
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (isinstance(target, ast.Attribute)
                    and target.attr == "implementation"
                    and not isinstance(target.value, ast.Name)):
                offenders.append(ast.dump(target))
            elif (isinstance(target, ast.Attribute)
                  and target.attr == "implementation"
                  and isinstance(target.value, ast.Attribute)):
                offenders.append(ast.dump(target))
    assert offenders == [], offenders
    assert SI.defines_function(tree, "patched_factory")
    assert SI.defines_function(tree, "verify")
    assert "init_generalized_prospective_S5SSM(" in source
    assert "implementation=implementation" in source


def test_the_benchmark_reads_the_scan_back_rather_than_assuming_it_took():
    source = io.open(BENCHMARK).read()
    node = SI.function_node(SI.parse(BENCHMARK), "verify")
    body = ast.get_source_segment(source, node) or ""
    assert "keywords" in body and "raise SystemExit" in body
    # and the row carries the evidence, not the intention
    assert '"implementation_in_force"' in source
    assert '"implementation_requested"' in source
    # identical peak memory across arms is called out
    assert "identical_peak_memory_suspicious" in source
