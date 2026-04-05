from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd



def certification_power_table(turns: pd.DataFrame) -> pd.DataFrame:
    required = {
        "selector_id",
        "persona_name",
        "persona_pressure",
        "A_t",
        "gap_est",
        "vli_cert",
        "meta__oracle_est_mean",
        "meta__oracle_est_lcb",
        "meta__selected_mean",
        "meta__selected_ucb",
    }
    missing = required.difference(turns.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = turns.copy()
    df["oracle_radius"] = df["meta__oracle_est_mean"] - df["meta__oracle_est_lcb"]
    df["selected_radius"] = df["meta__selected_ucb"] - df["meta__selected_mean"]
    df["radius_sum"] = df["oracle_radius"] + df["selected_radius"]
    df["cert_margin"] = df["gap_est"] - df["radius_sum"]
    df["certifiable_by_margin"] = df["cert_margin"] > 0.0
    df["certified_active"] = df["vli_cert"] > 0.0
    if "gap_true" in df.columns:
        df["positive_true_gap"] = df["gap_true"] > 0.0
    else:
        df["positive_true_gap"] = df["gap_est"] > 0.0

    rows = []
    group_cols = ["selector_id", "persona_name", "persona_pressure"]
    for keys, group in df.groupby(group_cols, dropna=False, sort=False):
        valid = group[group["A_t"] == True]
        target = valid[valid["positive_true_gap"] == True]
        n_valid = int(len(valid))
        n_target = int(len(target))
        rows.append(
            {
                "selector_id": keys[0],
                "persona_name": keys[1],
                "persona_pressure": keys[2],
                "n_valid": n_valid,
                "n_positive_true_gap": n_target,
                "certification_rate_given_positive_true_gap": float(target["certified_active"].mean()) if n_target > 0 else np.nan,
                "margin_positive_rate_given_positive_true_gap": float(target["certifiable_by_margin"].mean()) if n_target > 0 else np.nan,
                "mean_cert_margin_given_positive_true_gap": float(target["cert_margin"].mean()) if n_target > 0 else np.nan,
                "median_cert_margin_given_positive_true_gap": float(target["cert_margin"].median()) if n_target > 0 else np.nan,
                "mean_radius_sum": float(valid["radius_sum"].mean()) if n_valid > 0 else np.nan,
                "mean_gap_est": float(valid["gap_est"].mean()) if n_valid > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)



def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose certification power from a turns.csv export.")
    parser.add_argument("--turns", required=True, help="Path to turns.csv")
    parser.add_argument("--out", default=None, help="Optional output CSV path")
    args = parser.parse_args()

    turns = pd.read_csv(args.turns)
    table = certification_power_table(turns)
    out_path = Path(args.out) if args.out else Path(args.turns).with_name("certification_power.csv")
    table.to_csv(out_path, index=False)
    print(f"[OK] wrote {out_path}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
