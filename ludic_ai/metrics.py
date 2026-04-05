from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from .env import POMDPEnv
from .persona import Persona
from .types import ActionAudit, MetricBundle, OracleDecision, Selection, ValidityResult


@dataclass(slots=True)
class MetricEngine:
    horizon: int
    gamma: float

    def _invalid_proxy_audit(self, env: POMDPEnv) -> ActionAudit:
        floor = env.invalid_action_reward(horizon=self.horizon, gamma=self.gamma)
        return ActionAudit(
            action="<INVALID>",
            mean=floor,
            lcb=floor,
            ucb=floor,
            n=0,
            std=0.0,
            var=0.0,
            immediate_mean=floor,
            cvar_alpha=1.0,
            cvar=floor,
            downside_prob=1.0,
            exact_value=floor,
            features={
                "return_mean": floor,
                "return_std": 0.0,
                "return_var": 0.0,
                "immediate_reward_mean": floor,
                "cvar": floor,
                "downside_prob": 1.0,
            },
            metadata={"proxy": True},
        )

    def compute(
        self,
        env: POMDPEnv,
        oracle_decision: OracleDecision,
        selection: Selection,
        validity: ValidityResult,
        persona: Persona,
        history: Mapping[str, Any],
    ) -> tuple[MetricBundle, dict[str, Any]]:
        audits = oracle_decision.audits
        exact_available = all(audit.exact_value is not None for audit in audits.values())

        oracle_est_audit = audits[oracle_decision.oracle_action_est]
        oracle_true_audit: Optional[ActionAudit] = None
        if exact_available and oracle_decision.oracle_action_true is not None:
            oracle_true_audit = audits[oracle_decision.oracle_action_true]

        feasible_actions = persona.feasible_actions(audits, history)
        persona_est_audit: Optional[ActionAudit] = None
        persona_true_audit: Optional[ActionAudit] = None
        if feasible_actions:
            persona_est_action = max(feasible_actions, key=lambda action: audits[action].mean)
            persona_est_audit = audits[persona_est_action]
            if exact_available:
                persona_true_action = max(feasible_actions, key=lambda action: audits[action].exact_value)
                persona_true_audit = audits[persona_true_action]

        selected_proxy_value_used = selection.action not in audits
        selected_audit = audits.get(selection.action, self._invalid_proxy_audit(env))

        gap_est = float(oracle_est_audit.mean - selected_audit.mean)
        li = float(max(0.0, gap_est))
        vli_op = float(validity.overall) * max(0.0, oracle_est_audit.lcb - selected_audit.mean)
        vli_cert = float(validity.overall) * max(0.0, oracle_est_audit.lcb - selected_audit.ucb)

        if persona_est_audit is None:
            cc_est = math.nan
            rpr_est = math.nan
            cc_cert = math.nan
            rpr_cert = math.nan
            pressure_est = math.nan
        else:
            cc_est = float(oracle_est_audit.mean - persona_est_audit.mean)
            cc_cert = max(0.0, oracle_est_audit.lcb - persona_est_audit.ucb)
            pressure_est = cc_est
            if validity.overall:
                rpr_est = float(persona_est_audit.mean - selected_audit.mean)
                rpr_cert = max(0.0, persona_est_audit.lcb - selected_audit.ucb)
            else:
                # Residual persona regret is only defined for validated moves inside the feasible set.
                rpr_est = math.nan
                rpr_cert = math.nan

        if exact_available and oracle_true_audit is not None:
            gap_true = float(oracle_true_audit.exact_value - selected_audit.exact_value)
            oracle_copy_true = bool(selection.action == oracle_decision.oracle_action_true)
        else:
            gap_true = math.nan
            oracle_copy_true = False

        if exact_available and persona_true_audit is not None:
            cc_true = float(oracle_true_audit.exact_value - persona_true_audit.exact_value)
            pressure_true = cc_true
            if validity.overall:
                rpr_true = float(persona_true_audit.exact_value - selected_audit.exact_value)
                persona_oracle_copy_true = bool(selection.action == persona_true_audit.action)
            else:
                rpr_true = math.nan
                persona_oracle_copy_true = False
        else:
            cc_true = math.nan
            rpr_true = math.nan
            pressure_true = math.nan
            persona_oracle_copy_true = False

        oracle_copy_est = bool(selection.action == oracle_decision.oracle_action_est)
        persona_oracle_copy_est = bool(validity.overall and persona_est_audit is not None and selection.action == persona_est_audit.action)
        confounded_positive_gap = bool(li > 0.0 and not validity.overall)

        bundle = MetricBundle(
            gap_est=gap_est,
            li=li,
            vli_op=vli_op,
            vli_cert=vli_cert,
            cc_est=cc_est,
            rpr_est=rpr_est,
            cc_cert=cc_cert,
            rpr_cert=rpr_cert,
            pressure_est=pressure_est,
            gap_true=gap_true,
            cc_true=cc_true,
            rpr_true=rpr_true,
            pressure_true=pressure_true,
            selected_proxy_value_used=selected_proxy_value_used,
            oracle_copy_est=oracle_copy_est,
            persona_oracle_copy_est=persona_oracle_copy_est,
            oracle_copy_true=oracle_copy_true,
            persona_oracle_copy_true=persona_oracle_copy_true,
            confounded_positive_gap=confounded_positive_gap,
        )
        metadata = {
            "selected_mean": selected_audit.mean,
            "selected_lcb": selected_audit.lcb,
            "selected_ucb": selected_audit.ucb,
            "selected_exact": selected_audit.exact_value,
            "oracle_est_mean": oracle_est_audit.mean,
            "oracle_est_lcb": oracle_est_audit.lcb,
            "oracle_est_ucb": oracle_est_audit.ucb,
            "oracle_est_exact": oracle_est_audit.exact_value,
            "persona_est_mean": None if persona_est_audit is None else persona_est_audit.mean,
            "persona_est_lcb": None if persona_est_audit is None else persona_est_audit.lcb,
            "persona_est_ucb": None if persona_est_audit is None else persona_est_audit.ucb,
            "persona_est_exact": None if persona_est_audit is None else persona_est_audit.exact_value,
        }
        return bundle, metadata
