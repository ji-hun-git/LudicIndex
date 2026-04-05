#!/usr/bin/env python3
"""Generate manuscript figures from exported Ludic AI CSV outputs.

Usage:
  python scripts/plot_publication_figures.py \
      --results_dir results/reference_outputs \
      --out_dir paper/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_confound(confound_df: pd.DataFrame, out_dir: Path) -> None:
    grounded = confound_df[confound_df["selector_id"] != "free_form_random"].copy()
    free = confound_df[confound_df["selector_id"] == "free_form_random"].copy()

    labels = ["free_form_random", "grounded_cap_low", "grounded_cap_mid", "grounded_cap_high"]
    values = []
    for label in labels:
        sub = confound_df[confound_df["selector_id"] == label]
        values.append(float(sub["confound_rate"].mean()) if len(sub) else np.nan)

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.bar(labels, values)
    ax.set_ylabel("Mean ConfoundRate")
    ax.set_xlabel("Selector")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Epistemic disentanglement")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_dir / "confound_collapse.pdf")
    fig.savefig(out_dir / "confound_collapse.png", dpi=200)
    plt.close(fig)


def plot_cc_vs_pressure(state_bank_episodes: pd.DataFrame, out_dir: Path) -> None:
    sub = state_bank_episodes[
        state_bank_episodes["selector_id"].astype(str).str.startswith("grounded_cap")
        & state_bank_episodes["persona_name"].isin(["risk_seeking", "loss_averse"])
    ].copy()
    grouped = (
        sub.groupby(["persona_name", "persona_pressure"], dropna=False)["sum_cc_true"]
        .mean()
        .reset_index()
        .sort_values(["persona_name", "persona_pressure"])
    )

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for persona_name, g in grouped.groupby("persona_name"):
        ax.plot(g["persona_pressure"], g["sum_cc_true"], marker="o", label=persona_name)
    ax.set_xlabel("Persona pressure")
    ax.set_ylabel("Mean cumulative constraint cost")
    ax.set_title("Constraint cost rises with pressure")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "cc_vs_pressure.pdf")
    fig.savefig(out_dir / "cc_vs_pressure.png", dpi=200)
    plt.close(fig)


def plot_rpr_vs_capability(state_bank_episodes: pd.DataFrame, out_dir: Path) -> None:
    sub = state_bank_episodes[
        state_bank_episodes["selector_id"].astype(str).str.startswith("grounded_cap")
        & state_bank_episodes["persona_name"].isin(["neutral", "risk_seeking", "build_preserving"])
    ].copy()
    grouped = (
        sub.groupby(["persona_name", "selector_capability"], dropna=False)["sum_rpr_true"]
        .mean()
        .reset_index()
        .sort_values(["persona_name", "selector_capability"])
    )

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for persona_name, g in grouped.groupby("persona_name"):
        ax.plot(g["selector_capability"], g["sum_rpr_true"], marker="o", label=persona_name)
    ax.set_xlabel("Selector capability")
    ax.set_ylabel("Mean cumulative residual persona regret")
    ax.set_title("Gated residual persona regret falls with capability")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "rpr_vs_capability.pdf")
    fig.savefig(out_dir / "rpr_vs_capability.png", dpi=200)
    plt.close(fig)


def plot_pressure_response(turns_df: pd.DataFrame, out_dir: Path) -> None:
    grounded = turns_df[turns_df["selector_id"].astype(str).str.startswith("grounded_cap")].copy()
    grounded = grounded.dropna(subset=["pressure_true"])
    if len(grounded) == 0:
        return
    grounded["pressure_bin"] = pd.cut(grounded["pressure_true"], bins=min(6, max(2, grounded["pressure_true"].nunique())), duplicates="drop")
    grouped = (
        grounded.groupby(["selector_capability", "pressure_bin"], observed=False)["A_t"]
        .mean()
        .reset_index()
    )
    grouped["pressure_mid"] = grouped["pressure_bin"].apply(lambda x: float(x.mid) if pd.notna(x) else np.nan)

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for capability, g in grouped.groupby("selector_capability"):
        g = g.sort_values("pressure_mid")
        ax.plot(g["pressure_mid"], g["A_t"], marker="o", label=f"cap={capability:g}")
    ax.set_xlabel("Pressure (binned true constraint cost)")
    ax.set_ylabel("Mean gate validity")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Pressure-response validity curve")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "pressure_response.pdf")
    fig.savefig(out_dir / "pressure_response.png", dpi=200)
    plt.close(fig)


def write_reference_summary_tex(results_dir: Path, out_dir: Path) -> None:
    confound = pd.read_csv(results_dir / "confound_rates.csv")
    state_eps = pd.read_csv(results_dir / "state_bank" / "episodes.csv")

    free = confound[confound["selector_id"] == "free_form_random"]
    grounded = confound[confound["selector_id"] != "free_form_random"]

    risk = state_eps[
        state_eps["selector_id"].astype(str).str.startswith("grounded_cap")
        & (state_eps["persona_name"] == "risk_seeking")
    ].groupby("persona_pressure")["sum_cc_true"].mean().sort_index()
    build = state_eps[
        state_eps["selector_id"].astype(str).str.startswith("grounded_cap")
        & (state_eps["persona_name"] == "build_preserving")
    ].groupby("selector_capability")["sum_rpr_true"].mean().sort_index()

    tex = f"""
% Auto-generated from scripts/plot_publication_figures.py
\\begin{{tabular}}{{p{{0.35\\linewidth}}p{{0.55\\linewidth}}}}
\\toprule
Diagnostic & Reference result \\\\
\\midrule
Free-form confound range & {free['confound_rate'].min():.2f}--{free['confound_rate'].max():.2f} \\\\
Grounded confound maximum & {grounded['confound_rate'].max():.2f} \\\\
Risk-seeking mean $\\sum \\CC_t$ & {risk.iloc[0]:.2f} $\\rightarrow$ {risk.iloc[-1]:.2f} \\\\
Build-preserving mean $\\sum \\RPR_t$ & {build.iloc[0]:.2f} $\\rightarrow$ {build.iloc[-1]:.2f} \\\\
\\bottomrule
\\end{{tabular}}
""".strip() + "\n"
    (out_dir / "reference_summary_table.tex").write_text(tex, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()

    _ensure_dir(args.out_dir)
    confound_df = pd.read_csv(args.results_dir / "confound_rates.csv")
    turns_df = pd.read_csv(args.results_dir / "turns.csv")
    state_bank_episodes = pd.read_csv(args.results_dir / "state_bank" / "episodes.csv")

    plot_confound(confound_df, args.out_dir)
    plot_cc_vs_pressure(state_bank_episodes, args.out_dir)
    plot_rpr_vs_capability(state_bank_episodes, args.out_dir)
    plot_pressure_response(turns_df, args.out_dir)
    write_reference_summary_tex(args.results_dir, args.out_dir.parent / "generated")


if __name__ == "__main__":
    main()
