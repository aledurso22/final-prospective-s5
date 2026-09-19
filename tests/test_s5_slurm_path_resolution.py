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
