#!/usr/bin/env python3
"""Plot knee15 prosthesis distillation loss curves for comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RUNS = [
    ("BC pdres (972)", "outputs/prosthesis_distill_knee15_pdres/2026-06-13/19-10-46/loss_curve.csv"),
    ("BC kit3 focus", "outputs/prosthesis_distill_knee15_kit3_focus/2026-06-13/20-11-40/loss_curve.csv"),
    ("DAgger r1", "outputs/prosthesis_distill_knee15_dagger_r1/2026-06-13/19-56-33/loss_curve.csv"),
    ("smooth train (dagger r1)", "outputs/prosthesis_distill_knee15_kit3_smooth_train/2026-06-13/20-28-50/loss_curve.csv"),
    ("smooth r2 (dagger r1+r2)", "outputs/prosthesis_distill_knee15_kit3_smooth_r2/2026-06-13/20-37-01/loss_curve.csv"),
    ("smooth r3", "outputs/prosthesis_distill_knee15_kit3_smooth_r3/2026-06-13/20-38-39/loss_curve.csv"),
    ("smooth r4", "outputs/prosthesis_distill_knee15_kit3_smooth_r4/2026-06-13/20-49-57/loss_curve.csv"),
    ("force tau r1", "outputs/prosthesis_distill_knee15_kit3_force_tau_r1/2026-06-13/21-03-45/loss_curve.csv"),
    ("teacherqfrc tau r1", "outputs/prosthesis_distill_knee15_kit3_teacherqfrc_tau_r1/2026-06-13/21-05-48/loss_curve.csv"),
    ("force equiv r1 (best)", "outputs/prosthesis_distill_knee15_kit3_force_equiv_r1/2026-06-13/21-36-46/loss_curve.csv"),
]

METRICS = [
    ("total_loss", "Train total loss"),
    ("val_loss", "Val total loss"),
    ("muscle_mse", "Train muscle MSE"),
    ("prosthesis_mse", "Train prosthesis MSE"),
    ("prosthesis_tau_mse", "Train prosthesis tau MSE"),
]


def load_runs(root: Path) -> list[tuple[str, pd.DataFrame]]:
    loaded: list[tuple[str, pd.DataFrame]] = []
    for label, rel_path in RUNS:
        path = root / rel_path
        if not path.exists():
            print(f"skip missing: {path}")
            continue
        df = pd.read_csv(path)
        loaded.append((label, df))
    return loaded


def plot_panel(
    runs: list[tuple[str, pd.DataFrame]],
    metric: str,
    title: str,
    ax: plt.Axes,
    *,
    log_y: bool = False,
) -> None:
    for label, df in runs:
        if metric not in df.columns:
            continue
        ax.plot(df["epoch"], df[metric], linewidth=1.8, label=label)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)
    if log_y:
        ax.set_yscale("log")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="musclemimic repo root",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/_nonformal_runs/knee15_loss_curve_plots"),
    )
    args = parser.parse_args()

    runs = load_runs(args.root)
    if not runs:
        raise SystemExit("No loss_curve.csv files found.")

    out_dir = args.root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Overview: total + val loss
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    plot_panel(runs, "total_loss", "Train total loss", axes[0])
    plot_panel(runs, "val_loss", "Val total loss", axes[1])
    axes[1].legend(fontsize=7, loc="upper right")
    fig.suptitle("Knee15 prosthesis distillation — total loss", fontsize=12)
    overview_path = out_dir / "knee15_distill_total_loss.png"
    fig.savefig(overview_path, dpi=160)
    plt.close(fig)

    # Component losses (linear scale)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    plot_panel(runs, "muscle_mse", "Train muscle MSE", axes[0, 0])
    plot_panel(runs, "prosthesis_mse", "Train prosthesis MSE", axes[0, 1])
    plot_panel(runs, "prosthesis_tau_mse", "Train prosthesis tau MSE", axes[1, 0])
    plot_panel(runs, "val_muscle_mse", "Val muscle MSE", axes[1, 1])
    axes[0, 0].legend(fontsize=6.5, loc="upper right")
    fig.suptitle("Knee15 prosthesis distillation — component losses", fontsize=12)
    components_path = out_dir / "knee15_distill_component_loss.png"
    fig.savefig(components_path, dpi=160)
    plt.close(fig)

    # Tau loss log scale (early runs have huge values)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    plot_panel(runs, "prosthesis_tau_mse", "Train prosthesis tau MSE (log)", ax, log_y=True)
    ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("Knee15 prosthesis distillation — tau MSE (log scale)", fontsize=12)
    tau_path = out_dir / "knee15_distill_tau_loss_log.png"
    fig.savefig(tau_path, dpi=160)
    plt.close(fig)

    # DAgger-focused subset
    dagger_labels = {
        "DAgger r1",
        "smooth train (dagger r1)",
        "smooth r2 (dagger r1+r2)",
        "smooth r3",
        "smooth r4",
        "force tau r1",
        "teacherqfrc tau r1",
        "force equiv r1 (best)",
    }
    dagger_runs = [(label, df) for label, df in runs if label in dagger_labels]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    plot_panel(dagger_runs, "total_loss", "Train total loss", axes[0])
    plot_panel(dagger_runs, "val_loss", "Val total loss", axes[1])
    axes[1].legend(fontsize=7, loc="upper right")
    fig.suptitle("Knee15 DAgger training runs — total loss", fontsize=12)
    dagger_path = out_dir / "knee15_dagger_total_loss.png"
    fig.savefig(dagger_path, dpi=160)
    plt.close(fig)

    print("Saved:")
    for path in (overview_path, components_path, tau_path, dagger_path):
        print(f"  {path}")


if __name__ == "__main__":
    main()
