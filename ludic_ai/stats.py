from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import scipy.stats as sps
import statsmodels.api as sm
import statsmodels.formula.api as smf


def paired_randomization_test(
    x: Sequence[float],
    y: Sequence[float],
    n_permutations: int = 10000,
    alternative: str = "greater",
    seed: Optional[int] = None,
) -> dict[str, float]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.shape != y_arr.shape:
        raise ValueError("x and y must have the same shape for a paired randomization test.")
    diffs = x_arr - y_arr
    observed = float(np.mean(diffs))
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, len(diffs)))
    permuted = np.mean(signs * diffs[None, :], axis=1)
    if alternative == "greater":
        p = float((np.sum(permuted >= observed) + 1) / (n_permutations + 1))
    elif alternative == "less":
        p = float((np.sum(permuted <= observed) + 1) / (n_permutations + 1))
    else:
        p = float((np.sum(np.abs(permuted) >= abs(observed)) + 1) / (n_permutations + 1))
    return {"observed_mean_difference": observed, "p_value": p, "n": float(len(diffs))}


def welch_t_test(
    x: Sequence[float],
    y: Sequence[float],
    alternative: str = "two-sided",
) -> dict[str, float]:
    res = sps.ttest_ind(np.asarray(x, dtype=float), np.asarray(y, dtype=float), equal_var=False, nan_policy="omit")
    stat = float(res.statistic)
    p_two = float(res.pvalue)
    if alternative == "greater":
        p = p_two / 2.0 if stat >= 0 else 1.0 - p_two / 2.0
    elif alternative == "less":
        p = p_two / 2.0 if stat <= 0 else 1.0 - p_two / 2.0
    else:
        p = p_two
    return {"t_stat": stat, "p_value": p}


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return {"mean": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan")}
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boots[i] = np.mean(sample)
    lower = float(np.quantile(boots, alpha / 2.0))
    upper = float(np.quantile(boots, 1.0 - alpha / 2.0))
    return {"mean": float(np.mean(arr)), "ci_lower": lower, "ci_upper": upper}


@dataclass(slots=True)
class ExperimentAnalyzer:
    """Post-hoc statistics over turn- and episode-level exports."""

    neutral_persona_name: str = "neutral"

    def _choose_value_col(self, episodes: pd.DataFrame, preferred_true: str, fallback_est: str) -> str:
        if preferred_true in episodes.columns and episodes[preferred_true].notna().any():
            return preferred_true
        return fallback_est

    def confound_rate_table(self, turns: pd.DataFrame) -> pd.DataFrame:
        if turns.empty:
            return pd.DataFrame()
        rows: list[dict[str, Any]] = []
        group_cols = ["selector_id", "persona_name", "persona_pressure"]
        for keys, group in turns.groupby(group_cols, dropna=False, sort=False):
            positive = group["li"] > 0.0
            denom = int(positive.sum())
            rows.append(
                {
                    "selector_id": keys[0],
                    "persona_name": keys[1],
                    "persona_pressure": keys[2],
                    "positive_gap_count": denom,
                    "confounded_gap_count": int(group.loc[positive, "confounded_positive_gap"].sum()),
                    "confound_rate": float(group.loc[positive, "confounded_positive_gap"].mean()) if denom > 0 else 0.0,
                    "invalid_rate": float((~group["legal_ok"]).mean()),
                    "parse_failure_rate": float((~group["parse_ok"]).mean()),
                }
            )
        return pd.DataFrame(rows)

    def pressure_trend_table(self, episodes: pd.DataFrame, value_col: str = "sum_cc_true") -> pd.DataFrame:
        if episodes.empty or value_col not in episodes.columns:
            return pd.DataFrame()
        rows: list[dict[str, Any]] = []
        grouped = (
            episodes.groupby(["selector_id", "persona_name", "persona_pressure"], dropna=False)[value_col]
            .mean()
            .reset_index()
        )
        for (selector_id, persona_name), group in grouped.groupby(["selector_id", "persona_name"], dropna=False):
            group = group.dropna(subset=["persona_pressure", value_col])
            if len(group) < 2:
                continue
            rho, p = sps.spearmanr(group["persona_pressure"], group[value_col], nan_policy="omit")
            slope, intercept, r_value, p_lin, stderr = sps.linregress(group["persona_pressure"], group[value_col])
            rows.append(
                {
                    "selector_id": selector_id,
                    "persona_name": persona_name,
                    "value_col": value_col,
                    "spearman_rho": float(rho),
                    "spearman_p": float(p),
                    "ols_slope": float(slope),
                    "ols_p": float(p_lin),
                    "n_pressure_levels": int(len(group)),
                }
            )
        return pd.DataFrame(rows)

    def capability_trend_table(self, episodes: pd.DataFrame, value_col: str = "sum_rpr_true") -> pd.DataFrame:
        if episodes.empty or value_col not in episodes.columns:
            return pd.DataFrame()
        rows: list[dict[str, Any]] = []
        grouped = (
            episodes.groupby(["persona_name", "persona_pressure", "selector_capability"], dropna=False)[value_col]
            .mean()
            .reset_index()
        )
        grouped = grouped.dropna(subset=["selector_capability", value_col])
        for (persona_name, persona_pressure), group in grouped.groupby(["persona_name", "persona_pressure"], dropna=False):
            if len(group) < 2:
                continue
            rho, p = sps.spearmanr(group["selector_capability"], group[value_col], nan_policy="omit")
            slope, intercept, r_value, p_lin, stderr = sps.linregress(group["selector_capability"], group[value_col])
            rows.append(
                {
                    "persona_name": persona_name,
                    "persona_pressure": float(persona_pressure),
                    "value_col": value_col,
                    "spearman_rho": float(rho),
                    "spearman_p": float(p),
                    "ols_slope": float(slope),
                    "ols_p": float(p_lin),
                    "n_capability_levels": int(len(group)),
                }
            )
        return pd.DataFrame(rows)

    def paired_persona_tests(self, episodes: pd.DataFrame, value_col: str = "sum_vli_cert") -> pd.DataFrame:
        if episodes.empty or value_col not in episodes.columns:
            return pd.DataFrame()
        rows: list[dict[str, Any]] = []
        neutral = episodes[episodes["persona_name"] == self.neutral_persona_name]
        if neutral.empty:
            return pd.DataFrame()
        for selector_id, group in episodes.groupby("selector_id", dropna=False):
            base = neutral[neutral["selector_id"] == selector_id][["seed", value_col]].rename(columns={value_col: "base_value"})
            for (persona_name, persona_pressure), comp in group.groupby(["persona_name", "persona_pressure"], dropna=False):
                if persona_name == self.neutral_persona_name:
                    continue
                merged = comp[["seed", value_col]].merge(base, on="seed", how="inner")
                merged = merged.dropna(subset=[value_col, "base_value"])
                if merged.empty:
                    continue
                rand = paired_randomization_test(merged[value_col], merged["base_value"], alternative="greater")
                welch = welch_t_test(merged[value_col], merged["base_value"], alternative="greater")
                rows.append(
                    {
                        "selector_id": selector_id,
                        "persona_name": persona_name,
                        "persona_pressure": float(persona_pressure),
                        "value_col": value_col,
                        "n_pairs": int(len(merged)),
                        "mean_diff": rand["observed_mean_difference"],
                        "paired_randomization_p": rand["p_value"],
                        "welch_t": welch["t_stat"],
                        "welch_p": welch["p_value"],
                    }
                )
        return pd.DataFrame(rows)

    def pressure_response_glm(self, turns: pd.DataFrame) -> pd.DataFrame:
        if turns.empty:
            return pd.DataFrame()
        pressure_col = "pressure_true" if turns["pressure_true"].notna().any() else "pressure_est"
        data = turns.dropna(subset=[pressure_col]).copy()
        if data.empty:
            return pd.DataFrame()
        data["selector_capability"] = data["selector_capability"].fillna(0.0)
        model = smf.glm(
            formula=f"A_t ~ {pressure_col} + selector_capability + C(persona_name)",
            data=data,
            family=sm.families.Binomial(),
        )
        fit = model.fit()
        table = fit.summary2().tables[1].reset_index().rename(columns={"index": "term"})
        table["pressure_column"] = pressure_col
        return table

    def analyze(self, turns: pd.DataFrame, episodes: pd.DataFrame) -> dict[str, pd.DataFrame]:
        cc_value_col = self._choose_value_col(episodes, preferred_true="sum_cc_true", fallback_est="sum_cc_est")
        rpr_value_col = self._choose_value_col(episodes, preferred_true="sum_rpr_true", fallback_est="sum_rpr_est")
        return {
            "confound_rates": self.confound_rate_table(turns),
            "pressure_trends": self.pressure_trend_table(episodes, value_col=cc_value_col),
            "capability_trends": self.capability_trend_table(episodes, value_col=rpr_value_col),
            "paired_tests": self.paired_persona_tests(episodes, value_col="sum_vli_cert"),
            "pressure_response_glm": self.pressure_response_glm(turns),
        }
