"""2. The impossibility witness - constructed exactly, not searched for.

Two physical histories A and B with

    w^(A)_{1:t} = w^(B)_{1:t}   at theta = 0        (identical local evidence)
    b^(A)_t    != b^(B)_t                            (different hidden state)

The learning corrections they require, -b^(A)_t and -b^(B)_t, therefore differ
for the SAME locally observable prospective history.

Why the construction is trivial, and why that is the point: at theta = 0 the
b-subsystem is completely decoupled from s, so b leaves no trace whatsoever in
w. Any causal function of w_{1:t} - more taps, local recurrence, a deeper
network, any capacity at all - returns the same value on A and B, while the
correct answer differs. The information is absent, not merely unextracted.

This numerical witness ILLUSTRATES the analytical theorem; it does not prove it.
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.learning_access._common import (AQUA, BLUE, ORANGE, SURFACE,
                                                 YELLOW, outdir, save, style)
from prospective.three_state import (ThreeStateConfig, numpy_rollout,
                                     prospective_inverse, w_of)

SEED, T = 20260910, 30


def build_pair(seed=SEED, T=T):
    """Same (x, z0), different (b0, v). Identical w, different b."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=T)
    cfg_A = ThreeStateConfig(b0=-4.0)
    cfg_B = ThreeStateConfig(b0=+2.5)
    v_A = rng.normal(size=T)
    v_B = rng.normal(size=T) * 0.5 - 0.3
    return cfg_A, cfg_B, x, v_A, v_B


def main():
    d = outdir("2_impossibility_witness")
    cfg_A, cfg_B, x, v_A, v_B = build_pair()

    wA = np.asarray(w_of(cfg_A, 0.0, x, v_A))
    wB = np.asarray(w_of(cfg_B, 0.0, x, v_B))
    A = numpy_rollout(cfg_A, 0.0, x, v_A)
    B = numpy_rollout(cfg_B, 0.0, x, v_B)
    bA, bB = A["b"][1:], B["b"][1:]

    w_gap = float(np.max(np.abs(wA - wB)))
    b_gap = float(np.max(np.abs(bA - bB)))
    corr_gap = float(np.max(np.abs((-bA) - (-bB))))

    print(f"  identical local evidence : max |w_A - w_B| = {w_gap:.3e}")
    print(f"  different hidden state   : max |b_A - b_B| = {b_gap:.4f}")
    print(f"  required corrections     : max |(-b_A) - (-b_B)| = {corr_gap:.4f}")
    print(f"  at the final step t={T}: b_A = {bA[-1]:+.4f}, b_B = {bB[-1]:+.4f}")

    # Any causal function of w must agree on A and B, since the inputs agree.
    # Demonstrate with a family of local correctors of increasing capacity:
    # more taps cannot help, because every tap sees the same numbers.
    caps = {}
    for n_taps in (1, 2, 4, 8, 16):
        fa = wA[-n_taps:]
        fb = wB[-n_taps:]
        caps[n_taps] = float(np.max(np.abs(fa - fb)))
    print("\n  max difference in the last n taps of w (any local corrector's input):")
    for n, gv in caps.items():
        print(f"    n_taps={n:>3}: {gv:.3e}")
    print("  -> every local corrector receives identical input, so it must emit")
    print("     an identical correction, while the correct ones differ by "
          f"{corr_gap:.4f}")

    ok = w_gap < 1e-12 and b_gap > 1.0
    print(f"\n  VERDICT: {'PASS' if ok else 'FAIL'} - witness constructed")

    results = dict(max_abs_w_difference=w_gap, max_abs_b_difference=b_gap,
                   max_abs_correction_difference=corr_gap,
                   b_A_final=float(bA[-1]), b_B_final=float(bB[-1]),
                   tap_window_differences={str(k): v for k, v in caps.items()},
                   verdict_pass=bool(ok))

    k = np.arange(1, T + 1)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), facecolor=SURFACE)
    axes[0].plot(k, wA, color=AQUA, lw=3.0, label="history A")
    axes[0].plot(k, wB, color=ORANGE, lw=1.5, ls=(0, (3, 2)), label="history B")
    style(axes[0], "Observable prospective history: identical", "step", "w",
          legend=True)
    axes[0].text(0.96, 0.06, f"max diff = {w_gap:.1e}",
                 transform=axes[0].transAxes, ha="right", fontsize=8.5)

    axes[1].plot(k, bA, color=AQUA, lw=2.4, label="b, history A")
    axes[1].plot(k, bB, color=ORANGE, lw=2.4, ls=(0, (4, 2)), label="b, history B")
    style(axes[1], "Hidden physical state: different", "step", "b", legend=True)

    axes[2].plot(k, -bA, color=AQUA, lw=2.4, label="required correction, A")
    axes[2].plot(k, -bB, color=ORANGE, lw=2.4, ls=(0, (4, 2)),
                 label="required correction, B")
    style(axes[2], "Required learning corrections: different", "step",
          "-b", legend=True)

    fig.suptitle("Same local evidence, different required correction - no local "
                 "corrector can manufacture the missing information",
                 color="#0b0b0b", fontsize=12, x=0.008, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    path = f"{d}/fig_impossibility_witness.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE); plt.close(fig)

    print("wrote", save("2_impossibility_witness",
                        dict(seed=SEED, T=T, b0_A=cfg_A.b0, b0_B=cfg_B.b0),
                        results), "and", path)
    return results


if __name__ == "__main__":
    main()
