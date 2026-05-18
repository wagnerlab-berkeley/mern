"""Run the paper KL-search old/new repro check end to end.

This is the fast guardrail for the paper model settings. It runs the old
``scvi.external.mern`` implementation in ``scvi-mern-new`` and the merged
``mern`` package in ``mern-dev``, then compares compact JSON summaries.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "tests" / "repro_outputs" / "current"
DEFAULT_BASELINE_DIR = ROOT / "tests" / "repro_outputs" / "baselines"


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return ROOT / path


def _latest_baseline(baseline_dir: Path, filename: str) -> Path | None:
    matches = [path for path in baseline_dir.rglob(filename) if path.is_file()]
    if not matches:
        return None
    return max(matches, key=lambda path: (path.stat().st_mtime, str(path)))


def _save_baseline(baseline_dir: Path, label: str | None, outputs: list[Path]) -> None:
    if label is None:
        label = datetime.now().strftime("%Y%m%d-%H%M%S")
    destination = baseline_dir / label
    destination.mkdir(parents=True, exist_ok=True)
    for output in outputs:
        shutil.copy2(output, destination / output.name)
    print(f"Saved baseline outputs in {destination}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-env", default="scvi-mern-new")
    parser.add_argument("--new-env", default="mern-dev")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--save-baseline", action="store_true")
    parser.add_argument("--baseline-label", default=None)
    parser.add_argument("--rep", type=int, default=1)
    parser.add_argument("--max-kl-weight", type=float, default=0.001)
    parser.add_argument("--graph-kl-weight", type=float, default=0.1)
    parser.add_argument("--n-metabolic-dim", type=int, default=25)
    parser.add_argument("--n-background-dim", type=int, default=15)
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args()

    output_dir = _resolve_path(args.output_dir)
    baseline_dir = _resolve_path(args.baseline_dir)

    suffix = (
        f"rep{args.rep}_maxkl{args.max_kl_weight}_graphkl{args.graph_kl_weight}_"
        f"met{args.n_metabolic_dim}_"
        f"back{args.n_background_dim}_{'init' if args.skip_train else f'{args.max_epochs}epoch'}"
    )
    old_output = output_dir / f"old_mouse_intestine_kl_search_{suffix}.json"
    new_output = output_dir / f"new_mouse_intestine_kl_search_{suffix}.json"

    common = [
        "--rep",
        str(args.rep),
        "--max-kl-weight",
        str(args.max_kl_weight),
        "--graph-kl-weight",
        str(args.graph_kl_weight),
        "--n-metabolic-dim",
        str(args.n_metabolic_dim),
        "--n-background-dim",
        str(args.n_background_dim),
        "--max-epochs",
        str(args.max_epochs),
    ]
    if args.skip_train:
        common.append("--skip-train")

    _run(
        [
            "conda",
            "run",
            "-n",
            args.old_env,
            "python",
            "scripts/run_kl_search_args_parity.py",
            "--impl",
            "old",
            "--output",
            str(old_output),
            *common,
        ]
    )
    _run(
        [
            "conda",
            "run",
            "-n",
            args.new_env,
            "python",
            "scripts/run_kl_search_args_parity.py",
            "--impl",
            "new",
            "--output",
            str(new_output),
            *common,
        ]
    )
    _run(
        [
            "conda",
            "run",
            "-n",
            args.new_env,
            "python",
            "scripts/compare_repro_parity.py",
            str(old_output),
            str(new_output),
            "--label",
            "old package vs merged package",
            "--rtol",
            str(args.rtol),
            "--atol",
            str(args.atol),
        ]
    )

    if not args.skip_baseline:
        baseline = _latest_baseline(baseline_dir, new_output.name)
        if baseline is None:
            print(f"No saved baseline found in {baseline_dir}; skipped baseline comparison")
        else:
            _run(
                [
                    "conda",
                    "run",
                    "-n",
                    args.new_env,
                    "python",
                    "scripts/compare_repro_parity.py",
                    str(baseline),
                    str(new_output),
                    "--label",
                    f"latest saved baseline ({baseline.parent.name}) vs current merged package",
                    "--rtol",
                    str(args.rtol),
                    "--atol",
                    str(args.atol),
                ]
            )

    if args.save_baseline:
        _save_baseline(baseline_dir, args.baseline_label, [old_output, new_output])


if __name__ == "__main__":
    main()
