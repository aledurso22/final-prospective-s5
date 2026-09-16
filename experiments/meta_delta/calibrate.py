"""Declared initializations. Host-side, analytic; no data, no scores.

* adaptive_delta and gp_two_sided start at the SAME eta0 = -log(1 - beta*),
  the completed study's closed-form delta initialization (beta* its reference
  single-write query amplitude). The candidate adds tau0 = 1 and raw_r = 0, so
  it IS that delta model at the start.
* heavy_ball_same_mass starts at the candidate's gamma = 1/eta0 and
  M = tau0/eta0, with T Rdot removed. It is NOT calibrated to beta* and does
  not equal delta at the start; its initial single-write amplitude is recorded.
* tss_eq17 starts at T0 = 10 token intervals and eta0 chosen so its
  single-write, one-idle-interval query amplitude (1 + 2h/T0) eta w equals
  beta* at w = 1 (derived in dynamics.eq17_step's documentation).
* gated_delta and momentum_delta: their official initializers, unchanged.

Every family gets the SAME two development slots: outer learning rate 0.003
(A) and 0.01 (B), identical initialization otherwise. Equal and identical
selection axis.
"""

import math

from experiments.adaptive_memory import calibrate as AC

from . import dynamics as MD

TSS_T0 = 10.0
LRS = (("A", 0.003), ("B", 0.01))


def configurations():
    bs = AC.beta_star()[0]
    eta0 = -math.log(1.0 - bs)
    eq17_eta0 = bs / (1.0 + 2.0 * MD.H / TSS_T0)
    slots = {}
    for tag, lr in LRS:
        slots[f"gp_two_sided/{tag}"] = dict(
            rule="gp_two_sided", config=tag, lr=lr, raw_eta=math.log(eta0),
            raw_tau=math.log(MD.TAU0), raw_r=0.0, eta=eta0, tau=MD.TAU0,
            rho=1.0)
        slots[f"adaptive_delta/{tag}"] = dict(
            rule="adaptive_delta", config=tag, lr=lr, raw_eta=math.log(eta0),
            eta=eta0)
        slots[f"heavy_ball_same_mass/{tag}"] = dict(
            rule="heavy_ball_same_mass", config=tag, lr=lr,
            raw_eta=math.log(eta0), raw_tau=math.log(MD.TAU0), eta=eta0,
            tau=MD.TAU0)
        slots[f"tss_eq17/{tag}"] = dict(
            rule="tss_eq17", config=tag, lr=lr, raw_eta=math.log(eq17_eta0),
            raw_T=math.log(TSS_T0), eta=eq17_eta0, T=TSS_T0)
        for rule in MD.LITERATURE:
            slots[f"{rule}/{tag}"] = dict(rule=rule, config=tag, lr=lr)
    x0 = eta0 * MD.TAU0 * MD.GATE_BOUND_L
    return dict(beta_star=bs, eta0=eta0, tau0=MD.TAU0, eq17_T0=TSS_T0,
                eq17_eta0=eq17_eta0, slots=slots,
                initial_upper_rho=(x0 / (x0 - 1.0) if x0 > 1.0
                                   else float("inf")))
