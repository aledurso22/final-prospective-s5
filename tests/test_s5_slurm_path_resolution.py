import os
import subprocess
from pathlib import Path


def test_relocated_sbatch_body_uses_submit_root(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    source = (repo / "bin/slurm/s5_three_arm_full_array.sbatch").read_text()
    preamble = source[source.index("set -euo pipefail"):source.index(
        ': "${S5_THREE_ARM_DATA')]
    relocated = tmp_path / "slurm_script"
    relocated.write_text(preamble + '\nprintf "%s\\n" "$PROSPECTIVE_REPO"\n')
    relocated.chmod(0o755)
    environment = os.environ.copy()
    environment.pop("PROSPECTIVE_REPO", None)
    environment["SLURM_SUBMIT_DIR"] = str(repo)
    result = subprocess.run(["bash", str(relocated)], cwd=tmp_path,
                            env=environment, text=True,
                            capture_output=True, check=True)
    assert result.stdout.strip().splitlines()[-1] == str(repo)


def test_s5_slurm_scripts_do_not_resolve_from_script_location():
    repo = Path(__file__).resolve().parents[1]
    for name in ("s5_three_arm_full_array.sbatch",
                 "s5_three_arm_full_finalize.sbatch"):
        source = (repo / "bin/slurm" / name).read_text()
        assert 'dirname "${BASH_SOURCE[0]}"' not in source
        assert 'SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is unset' in source
        assert '--chdir="$REPO_ROOT"' in (
            (repo / "bin/run_experiments/cluster_s5_three_arm_full.sh").read_text())


def test_slurm_logs_are_under_run_output_not_repository():
    repo = Path(__file__).resolve().parents[1]
    launcher = (repo / "bin/run_experiments/cluster_s5_three_arm_full.sh").read_text()
    assert 'mkdir -p "$OUTPUT_DIR/slurm"' in launcher
    assert '--output="$OUTPUT_DIR/slurm/array-%A_%a.out"' in launcher
    assert '--error="$OUTPUT_DIR/slurm/array-%A_%a.err"' in launcher
    assert '--output="$OUTPUT_DIR/slurm/finalizer-%j.out"' in launcher
    assert '--error="$OUTPUT_DIR/slurm/finalizer-%j.err"' in launcher
    assert "--dependency=\"afterany:${ARRAY_JOB}\"" in launcher
    for name in ("s5_three_arm_full_array.sbatch",
                 "s5_three_arm_full_finalize.sbatch"):
        source = (repo / "bin/slurm" / name).read_text()
        assert "#SBATCH --output" not in source
        assert "#SBATCH --error" not in source


def test_smoke_uses_same_runner_and_writes_result_artifact():
    repo = Path(__file__).resolve().parents[1]
    source = (repo / "bin/slurm/s5_three_arm_full_array.sbatch").read_text()
    runner = (repo / "experiments/s5_three_arm_full/runner.py").read_text()
    assert "S5_THREE_ARM_SMOKE" in source
    assert "--smoke" in source
    assert '"smoke_result.json"' in runner
    assert '"SMOKE_PASS"' in runner
