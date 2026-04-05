#!/usr/bin/env python3
"""Print a concise textual summary of the main CSV outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=Path, required=True)
    args = parser.parse_args()

    confound = pd.read_csv(args.results_dir / "confound_rates.csv")
    episodes = pd.read_csv(args.results_dir / "state_bank" / "episodes.csv")

    free = confound[confound["selector_id"] == "free_form_random"]
    grounded = confound[confound["selector_id"] != "free_form_random"]
    print("Free-form confound range:", f"{free['confound_rate'].min():.3f} to {free['confound_rate'].max():.3f}")
    print("Grounded max confound:", f"{grounded['confound_rate'].max():.3f}")

    for persona in ["risk_seeking", "loss_averse"]:
        sub = episodes[
            episodes["selector_id"].astype(str).str.startswith("grounded_cap")
            & (episodes["persona_name"] == persona)
        ].groupby("persona_pressure")["sum_cc_true"].mean().sort_index()
        if len(sub) >= 2:
            print(f"{persona} CC means:", sub.to_dict())

    for persona in ["neutral", "build_preserving", "risk_seeking"]:
        sub = episodes[
            episodes["selector_id"].astype(str).str.startswith("grounded_cap")
            & (episodes["persona_name"] == persona)
        ].groupby("selector_capability")["sum_rpr_true"].mean().sort_index()
        if len(sub) >= 2:
            print(f"{persona} RPR means:", sub.to_dict())


if __name__ == "__main__":
    main()
