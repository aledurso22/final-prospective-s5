"""Parallel-scan realizations of the existing two-compartment recurrence.

THE EQUATION IS UNCHANGED. Everything here reproduces

    s_t = a1 s_{t-1} + a2 s_{t-2} + c1 @ x_t + c2 @ x_{t-1}

with zero prehistory, exactly as `scan_companion_sequential` computes it.
`generalized_coefficients`, `target_map`, `response_mass_gamma` and Native
S5 are untouched; this module only changes HOW the same recurrence is
evaluated.

WHY. The production run used `scan_companion_sequential`, a `lax.scan` over
16,000 tokens wrapped in `jax.checkpoint`. That is O(L) sequential steps
with a rematerialized backward pass, and it measured about 2.98 steps per
minute, projecting roughly 155 hours for the wave. It stayed finite over
the observed prefix, so what it demonstrates is an IMPLEMENTATION
bottleneck, not an equation-level instability.

TWO PARALLEL ROUTES.

1. COMPANION. `scan_companion` (already present) runs the affine
   associative scan over H = [[a1, a2], [1, 0]]. Correct, but it carries a
   2x2 matrix per token per mode: four complex multiplies and two adds per
   combine, against one multiply and one add for a scalar scan.

2. FACTORED, preferred. The characteristic polynomial is

       lambda^2 - a1 lambda - a2 = 0,      r1 + r2 = a1,   r1 r2 = -a2,

   so (1 - r1 z^-1)(1 - r2 z^-1) = 1 - a1 z^-1 - a2 z^-2 and the second
   order recurrence is two FIRST-ORDER ones in series,

       v_t = r1 v_{t-1} + d_t,
       s_t = r2 s_{t-1} + v_t,

   each of which is exactly the shape `s5/ssm.py`'s own `binary_operator`
   composes. No 2x2 block, no new scan primitive.

ROOTS, COMPUTED STABLY. The naive pair (a1 +- sqrt(a1^2 + 4 a2))/2 loses
the small root to cancellation whenever |a1| dominates the square root.
The standard remedy is used: form the root that adds CONSTRUCTIVELY,

       sign = +1 if Re(conj(a1) * sqrt(disc)) >= 0 else -1,
       major = (a1 + sign * sqrt(disc)) / 2,

then obtain the other from the PRODUCT, minor = -a2 / major, which never
subtracts. Ordering is deterministic: `major` is always the root of larger
magnitude, and the cascade is order-independent anyway, which is tested.

REPEATED ROOTS ARE EXACT, NOT SPECIAL-CASED. At disc = 0 the factorization
gives r1 = r2 and the cascade becomes v_t = r v_{t-1} + d_t,
s_t = r s_{t-1} + v_t, which is the Jordan realization of the repeated-root
recurrence and reproduces t r^{t-1} exactly. A partial-fraction route would
divide by (r1 - r2) and fail there; this one does not. Near-repeated roots
are equally safe because `minor` comes from the product rather than a
difference.

NO EIGENDECOMPOSITION IS USED ANYWHERE IN THE TRAINING PATH.
"""

import jax
import jax.numpy as np

from .discrete_recurrence import scan_companion, scan_companion_sequential
from .ssm import binary_operator

#: |major| below this is treated as "both roots are zero", which happens
#: only when a1 and a2 both vanish
ROOT_FLOOR = 1e-30


def discriminant(a1, a2):
    """a1^2 + 4 a2, the discriminant of lambda^2 - a1 lambda - a2."""
    return a1 * a1 + 4.0 * a2


def companion_roots(a1, a2):
    """(major, minor): the two roots, stably and deterministically.

    `major` is the root of larger magnitude, formed so that the square root
    ADDS to a1 rather than cancelling it. `minor` follows from the product
    r1 r2 = -a2, so it is never the difference of two close numbers.
    """
    root = np.sqrt(discriminant(a1, a2).astype(a1.dtype))
    # the sign that makes |a1 + sign*root| the larger of the two options
    aligned = (np.conj(a1) * root).real if np.iscomplexobj(a1) else a1 * root
    sign = np.where(aligned >= 0, 1.0, -1.0).astype(a1.dtype)
    major = (a1 + sign * root) / 2.0
    safe = np.where(np.abs(major) < ROOT_FLOOR,
                    np.ones_like(major), major)
    minor = np.where(np.abs(major) < ROOT_FLOOR,
                     np.zeros_like(major), -a2 / safe)
    return major, minor


def roots_to_coefficients(r1, r2):
    """The inverse map: a1 = r1 + r2, a2 = -r1 r2."""
    return r1 + r2, -(r1 * r2)


def spectral_radius_from_roots(a1, a2):
    """max(|r1|, |r2|), which equals `companion_radius` by construction."""
    major, minor = companion_roots(a1, a2)
    return np.maximum(np.abs(major), np.abs(minor))


def _drive(c1, c2, sequence):
    """c1 @ x_t + c2 @ x_{t-1}, zero prehistory.

    Byte-for-byte the expression `scan_companion` builds, so the parallel
    routes cannot differ from the oracle by the drive.
    """
    previous = np.concatenate((np.zeros_like(sequence[:1]), sequence[:-1]),
                              axis=0)
    return (jax.vmap(lambda x: c1 @ x)(sequence)
            + jax.vmap(lambda x: c2 @ x)(previous))


def _first_order(pole, drive):
    """u_t = pole * u_{t-1} + drive_t, through the repository's own scan."""
    poles = pole * np.ones((drive.shape[0],) + pole.shape, dtype=drive.dtype)
    _, states = jax.lax.associative_scan(binary_operator, (poles, drive))
    return states


def scan_factored(a1, a2, c1, c2, inputs, reverse=False):
    """The same recurrence, as two scalar associative scans."""
    sequence = inputs[::-1] if reverse else inputs
    drive = _drive(c1, c2, sequence).astype(a1.dtype)
    first, second = companion_roots(a1, a2)
    states = _first_order(second, _first_order(first, drive))
    return states[::-1] if reverse else states


# --------------------------------------------- scans over a GIVEN drive ---
# The scans above build the drive from (c1, c2) internally. The matched-lag
# arm forms its drive differently -- regrouped, to avoid a cancellation --
# so it needs the same recurrences over a drive it supplies itself. The
# LHS is identical in every case: s_t = a1 s_{t-1} + a2 s_{t-2} + d_t.


def scan_factored_drive(a1, a2, drive):
    """The two-scalar-scan form over a precomputed drive."""
    first, second = companion_roots(a1, a2)
    return _first_order(second, _first_order(first, drive.astype(a1.dtype)))


def scan_sequential_drive(a1, a2, drive):
    """The oracle over a precomputed drive, rematerialized as before."""
    def rollout(values):
        def body(carry, value):
            state, previous = carry
            current = a1 * state + a2 * previous + value
            return (current, state), current

        initial = (np.zeros_like(a1), np.zeros_like(a1))
        _, states = jax.lax.scan(body, initial, values)
        return states

    return jax.checkpoint(rollout)(drive.astype(a1.dtype))


DRIVE_SCANS = {"sequential": scan_sequential_drive,
               "factored": scan_factored_drive}


def drive_scan_for(implementation):
    if implementation not in DRIVE_SCANS:
        raise ValueError(
            f"unknown drive scan {implementation!r}; "
            f"known: {sorted(DRIVE_SCANS)}")
    return DRIVE_SCANS[implementation]


#: the sequential implementation stays the DEFAULT and the oracle. The
#: alternatives are opt-in by name, never by silent substitution.
SCAN_IMPLEMENTATIONS = {
    "sequential": scan_companion_sequential,
    "companion": scan_companion,
    "factored": scan_factored,
}
DEFAULT_IMPLEMENTATION = "sequential"


def scan_for(implementation):
    """Look up a scan by name, refusing an unknown one loudly."""
    if implementation not in SCAN_IMPLEMENTATIONS:
        raise ValueError(
            f"unknown scan implementation {implementation!r}; "
            f"known: {sorted(SCAN_IMPLEMENTATIONS)}")
    return SCAN_IMPLEMENTATIONS[implementation]
