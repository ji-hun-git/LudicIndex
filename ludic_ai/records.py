from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

from .types import EpisodeSummary, TurnRecord


@dataclass
class TurnLog:
    turns: list[TurnRecord] = field(default_factory=list)

    def add(self, record: TurnRecord) -> None:
        self.turns.append(record)

    def extend(self, records: Iterable[TurnRecord]) -> None:
        self.turns.extend(records)

    def turn_frame(self) -> pd.DataFrame:
        if not self.turns:
            return pd.DataFrame()
        return pd.DataFrame([record.to_flat_dict() for record in self.turns])

    @staticmethod
    def _sum_preserve_nan(series: pd.Series) -> float:
        valid = series.dropna()
        if valid.empty:
            return float("nan")
        return float(valid.sum())

    @staticmethod
    def _mean_preserve_nan(series: pd.Series) -> float:
        valid = series.dropna()
        if valid.empty:
            return float("nan")
        return float(valid.mean())

    def episode_frame(self) -> pd.DataFrame:
        turns = self.turn_frame()
        if turns.empty:
            return pd.DataFrame()
        group_cols = [
            "experiment_id",
            "run_id",
            "episode_id",
            "seed",
            "selector_id",
            "selector_capability",
            "persona_name",
            "persona_pressure",
        ]
        rows: list[dict] = []
        for keys, group in turns.groupby(group_cols, dropna=False, sort=False):
            positive_gap = group["li"] > 0.0
            confound_rate = (
                float(group.loc[positive_gap, "confounded_positive_gap"].mean()) if positive_gap.any() else 0.0
            )
            summary = EpisodeSummary(
                experiment_id=keys[0],
                run_id=keys[1],
                episode_id=keys[2],
                seed=None if pd.isna(keys[3]) else int(keys[3]),
                selector_id=keys[4],
                selector_capability=None if pd.isna(keys[5]) else float(keys[5]),
                persona_name=keys[6],
                persona_pressure=float(keys[7]),
                # Preserve NaN for matched-state banks or fully unexecuted episodes.
                episode_return=self._sum_preserve_nan(group["reward"]),
                n_turns=int(len(group)),
                invalid_rate=float((~group["legal_ok"]).mean()),
                adherence_rate=float(group["A_t"].mean()),
                oracle_copy_rate_est=float(group["oracle_copy_est"].mean()),
                persona_oracle_rate_est=float(group["persona_oracle_copy_est"].mean()),
                confound_rate=confound_rate,
                sum_li=float(group["li"].sum()),
                sum_vli_op=float(group["vli_op"].sum()),
                sum_vli_cert=float(group["vli_cert"].sum()),
                # Preserve NaN when the feasible set collapses for the entire episode.
                sum_cc_est=self._sum_preserve_nan(group["cc_est"]),
                sum_rpr_est=self._sum_preserve_nan(group["rpr_est"]),
                sum_cc_true=self._sum_preserve_nan(group["cc_true"]),
                sum_rpr_true=self._sum_preserve_nan(group["rpr_true"]),
                mean_pressure_est=self._mean_preserve_nan(group["pressure_est"]),
                mean_pressure_true=self._mean_preserve_nan(group["pressure_true"]),
            )
            rows.append(summary.to_flat_dict())
        return pd.DataFrame(rows)

    def export(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        turns = self.turn_frame()
        episodes = self.episode_frame()
        turns.to_csv(path / "turns.csv", index=False)
        episodes.to_csv(path / "episodes.csv", index=False)
        with open(path / "turns.jsonl", "w", encoding="utf-8") as f:
            for record in self.turns:
                f.write(json.dumps(record.to_flat_dict(), ensure_ascii=False) + "\n")
