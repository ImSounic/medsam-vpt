"""SLURM array ranges match their config lists, referenced configs exist, and bash parses."""

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SLURM = REPO / "cka" / "slurm"


def _array_len(text: str) -> int:
    m = re.search(r"^#SBATCH --array=(\d+)-(\d+)$", text, re.M)
    assert m, "missing --array"
    return int(m.group(2)) - int(m.group(1)) + 1


def _bash_array(text: str, name: str) -> list[str]:
    m = re.search(rf"^{name}=\((.*?)\)", text, re.M | re.S)
    assert m, f"missing {name}=( ... )"
    return [t.strip().strip('"') for t in m.group(1).split() if t.strip()]


@pytest.mark.parametrize(
    "script,var,n",
    [
        ("accv_t1.sbatch", "CONFIGS", 4),
        ("accv_t2.sbatch", "CONFIGS", 9),
        ("accv_t4.sbatch", "CONFIGS", 2),
        ("accv_t5.sbatch", "CONFIGS", 4),
        ("accv_t6.sbatch", "CONFIGS", 18),
    ],
)
def test_train_arrays_match_configs(script, var, n):
    text = (SLURM / script).read_text()
    items = _bash_array(text, var)
    assert len(items) == n
    assert _array_len(text) == n
    for cfg in items:
        assert (REPO / "configs" / cfg).exists(), cfg


def test_t3_array_covers_two_prompt_levels():
    text = (SLURM / "accv_t3.sbatch").read_text()
    assert _bash_array(text, "PERTURBS") == ["0", "50"]
    assert _array_len(text) == 2
    assert (
        len(_bash_array(text, "CHECKPOINTS")) == 6
    )  # plus zero-shot handled separately
    assert "--drift" in text


@pytest.mark.parametrize(
    "script",
    [
        "accv_t1.sbatch",
        "accv_t1_eval.sbatch",
        "accv_t2.sbatch",
        "accv_t2_eval.sbatch",
        "accv_t3.sbatch",
        "accv_t4.sbatch",
        "accv_t4_eval.sbatch",
        "accv_t5.sbatch",
        "accv_t5_eval.sbatch",
        "accv_dmid.sbatch",
        "accv_t6.sbatch",
        "accv_t6_eval.sbatch",
        "accv_t7.sbatch",
        "submit_accv.sh",
        "sync_to_hpc.sh",
    ],
)
def test_bash_syntax(script):
    subprocess.run(["bash", "-n", str(SLURM / script)], check=True)


def test_submit_chain_dependencies():
    text = (SLURM / "submit_accv.sh").read_text()
    assert "afterany:$T1" in text and "afterany:$T2" in text
    assert text.index("accv_t1.sbatch") < text.index("accv_t2.sbatch")


def test_t4_eval_covers_pm0_sweep_and_encoder_only_dumps():
    text = (SLURM / "accv_t4_eval.sbatch").read_text()
    assert "results_cka_oodonly_late_l10_pm0" in text
    assert "lora_encoder_only_r28_all_seed0/best.pth" in text
    assert (
        "--drift" in text and "for PM in 0 50" in text and "runs_t3_pm${PM}.csv" in text
    )


def test_t5_eval_covers_seeds_and_retrained_pm0_tight_eval():
    text = (SLURM / "accv_t5_eval.sbatch").read_text()
    assert "checkpoints/runs_accv_t5/*/best.pth" in text
    assert "runs_accv_t5.csv" in text and "results_accv_t5" in text
    assert "runs_cka_oodonly_late/lora_cka_oodonly_late_l10_seed0/best.pth" in text
    assert "runs_cka_pm0_retrained.csv" in text


def test_t7_covers_budget_and_full_data_checkpoints():
    text = (SLURM / "accv_t7.sbatch").read_text()
    ck = _bash_array(text, "CHECKPOINTS")
    assert (
        len(ck) == 12
        and sum("runs_accv_t2" in c for c in ck) == 9
        and "--drift" in text
    )
