"""Render saved memory-audit JSON without checkpoints, JAX, data or training.

python -m experiments.s5_predictor_corrector.memory_pole_render \
  --report /path/memory_poles.json --out /path/new-figures
"""
import argparse
import json
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--report", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    # Fail before creating outputs if the rendering environment is incomplete.
    from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: F401
    from .memory_pole_plots import plot_report
    from .memory_impulses import plot_impulse_report
    report_path, out = Path(args.report).resolve(), Path(args.out).resolve()
    if out.exists():
        raise ValueError(f"refusing existing figure directory: {out}")
    report = json.loads(report_path.read_text())
    out.mkdir(parents=True, exist_ok=False)
    files = plot_report(report, out) + plot_impulse_report(report, out)
    with (out / "render_status.json").open("x") as f:
        json.dump({"status": "completed", "source_report": str(report_path), "files": files}, f, indent=2)
    print(f"Rendered {len(files)} figures from saved numerical results into {out}")
    return files


if __name__ == "__main__":
    main()
