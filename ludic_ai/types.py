from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Hashable, Mapping, Optional, Sequence


Action = Hashable



def safe_repr(value: Any) -> str:
    try:
        return repr(value)
    except Exception:
        return f"<{type(value).__name__}>"


@dataclass(slots=True)
class StepResult:
    observation: Any
    reward: float
    terminated: bool
    truncated: bool = False
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ActionAudit:
    action: Action
    mean: float
    lcb: float
    ucb: float
    n: int
    std: float
    var: float
    immediate_mean: float
    cvar_alpha: float
    cvar: float
    downside_prob: float
    exact_value: Optional[float] = None
    rollout_returns: Optional[tuple[float, ...]] = None
    features: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def flat(self, prefix: str = "") -> dict[str, Any]:
        payload = {
            f"{prefix}action": safe_repr(self.action),
            f"{prefix}mean": self.mean,
            f"{prefix}lcb": self.lcb,
            f"{prefix}ucb": self.ucb,
            f"{prefix}n": self.n,
            f"{prefix}std": self.std,
            f"{prefix}var": self.var,
            f"{prefix}immediate_mean": self.immediate_mean,
            f"{prefix}cvar": self.cvar,
            f"{prefix}downside_prob": self.downside_prob,
            f"{prefix}exact_value": self.exact_value,
        }
        for key, value in self.features.items():
            payload[f"{prefix}feature__{key}"] = value
        return payload


@dataclass(slots=True)
class OracleDecision:
    state_id: str
    observation: Any
    legal_actions: tuple[Action, ...]
    menu_actions: tuple[Action, ...]
    audits: dict[Action, ActionAudit]
    horizon: int
    gamma: float
    oracle_action_est: Action
    oracle_action_true: Optional[Action] = None
    persona_action_est: Optional[Action] = None
    persona_action_true: Optional[Action] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Selection:
    action: Optional[Action]
    raw_output: Any = None
    parsed_ok: bool = True
    selector_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ValidityResult:
    parse_ok: bool
    menu_ok: bool
    legal_ok: bool
    persona_ok: bool
    overall: bool
    reasons: tuple[str, ...] = ()


@dataclass(slots=True)
class MetricBundle:
    gap_est: float
    li: float
    vli_op: float
    vli_cert: float
    cc_est: float
    rpr_est: float
    cc_cert: float
    rpr_cert: float
    pressure_est: float
    gap_true: float
    cc_true: float
    rpr_true: float
    pressure_true: float
    selected_proxy_value_used: bool
    oracle_copy_est: bool
    persona_oracle_copy_est: bool
    oracle_copy_true: bool
    persona_oracle_copy_true: bool
    confounded_positive_gap: bool


@dataclass(slots=True)
class TurnRecord:
    experiment_id: str
    run_id: str
    episode_id: str
    step_index: int
    seed: Optional[int]
    selector_id: str
    selector_capability: Optional[float]
    selector_uses_oracle: bool
    persona_name: str
    persona_pressure: float
    state_id: str
    selected_action: Optional[Action]
    oracle_action_est: Action
    oracle_action_true: Optional[Action]
    persona_action_est: Optional[Action]
    persona_action_true: Optional[Action]
    parse_ok: bool
    menu_ok: bool
    legal_ok: bool
    persona_ok: bool
    A_t: bool
    gap_est: float
    li: float
    vli_op: float
    vli_cert: float
    cc_est: float
    rpr_est: float
    cc_cert: float
    rpr_cert: float
    pressure_est: float
    gap_true: float
    cc_true: float
    rpr_true: float
    pressure_true: float
    oracle_copy_est: bool
    persona_oracle_copy_est: bool
    oracle_copy_true: bool
    persona_oracle_copy_true: bool
    confounded_positive_gap: bool
    selected_proxy_value_used: bool
    reward: float
    terminated: bool
    invalid_termination: bool
    menu_size: int
    legal_action_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_flat_dict(self) -> dict[str, Any]:
        payload = {
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "step_index": self.step_index,
            "seed": self.seed,
            "selector_id": self.selector_id,
            "selector_capability": self.selector_capability,
            "selector_uses_oracle": self.selector_uses_oracle,
            "persona_name": self.persona_name,
            "persona_pressure": self.persona_pressure,
            "state_id": self.state_id,
            "selected_action": safe_repr(self.selected_action),
            "oracle_action_est": safe_repr(self.oracle_action_est),
            "oracle_action_true": safe_repr(self.oracle_action_true),
            "persona_action_est": safe_repr(self.persona_action_est),
            "persona_action_true": safe_repr(self.persona_action_true),
            "parse_ok": self.parse_ok,
            "menu_ok": self.menu_ok,
            "legal_ok": self.legal_ok,
            "persona_ok": self.persona_ok,
            "A_t": self.A_t,
            "gap_est": self.gap_est,
            "li": self.li,
            "vli_op": self.vli_op,
            "vli_cert": self.vli_cert,
            "cc_est": self.cc_est,
            "rpr_est": self.rpr_est,
            "cc_cert": self.cc_cert,
            "rpr_cert": self.rpr_cert,
            "pressure_est": self.pressure_est,
            "gap_true": self.gap_true,
            "cc_true": self.cc_true,
            "rpr_true": self.rpr_true,
            "pressure_true": self.pressure_true,
            "oracle_copy_est": self.oracle_copy_est,
            "persona_oracle_copy_est": self.persona_oracle_copy_est,
            "oracle_copy_true": self.oracle_copy_true,
            "persona_oracle_copy_true": self.persona_oracle_copy_true,
            "confounded_positive_gap": self.confounded_positive_gap,
            "selected_proxy_value_used": self.selected_proxy_value_used,
            "reward": self.reward,
            "terminated": self.terminated,
            "invalid_termination": self.invalid_termination,
            "menu_size": self.menu_size,
            "legal_action_count": self.legal_action_count,
        }
        for key, value in self.metadata.items():
            payload[f"meta__{key}"] = value
        return payload


@dataclass(slots=True)
class EpisodeSummary:
    experiment_id: str
    run_id: str
    episode_id: str
    seed: Optional[int]
    selector_id: str
    selector_capability: Optional[float]
    persona_name: str
    persona_pressure: float
    episode_return: float
    n_turns: int
    invalid_rate: float
    adherence_rate: float
    oracle_copy_rate_est: float
    persona_oracle_rate_est: float
    confound_rate: float
    sum_li: float
    sum_vli_op: float
    sum_vli_cert: float
    sum_cc_est: float
    sum_rpr_est: float
    sum_cc_true: float
    sum_rpr_true: float
    mean_pressure_est: float
    mean_pressure_true: float

    def to_flat_dict(self) -> dict[str, Any]:
        return asdict(self)
