"""Plumbing for the learning-access family. Reuses the validated palette."""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from experiments.tss_closure._common import (AQUA, BLUE, GRID, INK, INK2,  # noqa
                                             MUTED, ORANGE, SURFACE, YELLOW,
                                             style, _enc)

RESULTS_ROOT = "results/learning_access"


def outdir(name):
    d = os.path.join(RESULTS_ROOT, name)
    os.makedirs(d, exist_ok=True)
    return d


def save(name, config, results):
    d = outdir(name)
    p = os.path.join(d, "summary.json")
    with open(p, "w") as fh:
        json.dump({"config": config, "results": results}, fh, indent=2, default=_enc)
    return p
