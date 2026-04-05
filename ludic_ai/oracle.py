from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .env import POMDPEnv, RandomRolloutPolicy, RolloutPolicy
from .persona import Persona
from .types import Action, ActionAudit, OracleDecision


@dataclass(slots=True)
class RolloutOracle:
    horizon: int
    n_rollouts: int
    gamma: float = 1.0
    delta: float = 0.05
    top_k: Optional[int] = None
    rollout_policy: Optional[RolloutPolicy] = None
    cvar_alpha: float = 0.1
    store_rollout_returns: bool = False
    ensure_all_feasible_in_menu: bool = False

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive.")
        if self.n_rollouts <= 0:
            raise ValueError("n_rollouts must be positive.")
        if not 0.0 < self.delta < 1.0:
            raise ValueError("delta must be in (0, 1).")
        if not 0.0 < self.cvar_alpha <= 1.0:
            raise ValueError("cvar_alpha must be in (0, 1].")
        if self.rollout_policy is None:
            self.rollout_policy = RandomRolloutPolicy()

    def audit(
        self,
        env: POMDPEnv,
        persona: Optional[Persona] = None,
        history: Optional[Mapping[str, Any]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> OracleDecision:
        history = dict(history or env.history_context())
        rng = rng or np.random.default_rng()
        legal_actions = tuple(env.legal_actions())
        audits = {action: self._audit_action(env, action, legal_actions, rng) for action in legal_actions}
        return self._decision_from_audits(env, audits, persona=persona, history=history)

    def rebuild_decision(
        self,
        env: POMDPEnv,
        audits: Mapping[Action, ActionAudit],
        persona: Optional[Persona] = None,
        history: Optional[Mapping[str, Any]] = None,
    ) -> OracleDecision:
        history = dict(history or env.history_context())
        return self._decision_from_audits(env, dict(audits), persona=persona, history=history)

    def _decision_from_audits(
        self,
        env: POMDPEnv,
        audits: Mapping[Action, ActionAudit],
        persona: Optional[Persona],
        history: Mapping[str, Any],
    ) -> OracleDecision:
        if not audits:
            raise RuntimeError("Oracle cannot build a decision with zero audited actions.")
        oracle_action_est = max(audits, key=lambda action: audits[action].mean)
        exact_available = all(audit.exact_value is not None for audit in audits.values())
        oracle_action_true = max(audits, key=lambda action: audits[action].exact_value) if exact_available else None
        persona_action_est: Optional[Action] = None
        persona_action_true: Optional[Action] = None
        if persona is not None:
            feasible = persona.feasible_actions(audits, history)
            if feasible:
                persona_action_est = max(feasible, key=lambda action: audits[action].mean)
                if exact_available:
                    persona_action_true = max(feasible, key=lambda action: audits[action].exact_value)
        menu_actions = self._build_menu(
            audits=audits,
            oracle_action_est=oracle_action_est,
            persona_action_est=persona_action_est,
            persona=persona,
            history=history,
        )
        return OracleDecision(
            state_id=env.state_id(),
            observation=env.oracle_view(),
            legal_actions=tuple(audits.keys()),
            menu_actions=tuple(menu_actions),
            audits=dict(audits),
            horizon=self.horizon,
            gamma=self.gamma,
            oracle_action_est=oracle_action_est,
            oracle_action_true=oracle_action_true,
            persona_action_est=persona_action_est,
            persona_action_true=persona_action_true,
            metadata={
                "delta": self.delta,
                "rollout_policy": self.rollout_policy.name,
                "cvar_alpha": self.cvar_alpha,
                "bound": "empirical_bernstein",
            },
        )

    def _build_menu(
        self,
        audits: Mapping[Action, ActionAudit],
        oracle_action_est: Action,
        persona_action_est: Optional[Action],
        persona: Optional[Persona],
        history: Mapping[str, Any],
    ) -> list[Action]:
        ordered = sorted(audits, key=lambda action: audits[action].mean, reverse=True)
        if self.top_k is None or self.top_k >= len(ordered):
            menu = list(ordered)
        else:
            menu = list(ordered[: self.top_k])
        required = [oracle_action_est]
        if persona_action_est is not None:
            required.append(persona_action_est)
        if self.ensure_all_feasible_in_menu and persona is not None:
            required.extend(persona.feasible_actions(audits, history))
        seen: set[Action] = set(menu)
        for action in required:
            if action not in seen:
                menu.append(action)
                seen.add(action)
        menu.sort(key=lambda action: audits[action].mean, reverse=True)
        return menu

    def _audit_action(
        self,
        env: POMDPEnv,
        action: Action,
        legal_actions: Sequence[Action],
        rng: np.random.Generator,
    ) -> ActionAudit:
        returns: list[float] = []
        immediate_rewards: list[float] = []
        for _ in range(self.n_rollouts):
            total, immediate = self._rollout_once(env, action, rng)
            returns.append(total)
            immediate_rewards.append(immediate)

        ret = np.asarray(returns, dtype=float)
        mean = float(np.mean(ret))
        # Use sample variance / standard deviation to match the empirical Bernstein interval.
        if len(ret) >= 2:
            var = float(np.var(ret, ddof=1))
            std = float(np.std(ret, ddof=1))
        else:
            var = 0.0
            std = 0.0
        cvar = self._cvar(ret, self.cvar_alpha)
        downside_prob = float(np.mean(ret < mean))
        lcb, ucb = self._confidence_interval(ret, env, n_actions=len(legal_actions))
        features = {
            "return_mean": mean,
            "return_std": std,
            "return_var": var,
            "immediate_reward_mean": float(np.mean(immediate_rewards)),
            "cvar": cvar,
            "downside_prob": downside_prob,
        }
        features.update(env.extra_action_features(action, rollout_returns=ret.tolist(), immediate_rewards=immediate_rewards))
        exact_value = env.exact_action_value(action, self.horizon, self.gamma, self.rollout_policy)
        return ActionAudit(
            action=action,
            mean=mean,
            lcb=lcb,
            ucb=ucb,
            n=self.n_rollouts,
            std=std,
            var=var,
            immediate_mean=float(np.mean(immediate_rewards)),
            cvar_alpha=self.cvar_alpha,
            cvar=cvar,
            downside_prob=downside_prob,
            exact_value=exact_value,
            rollout_returns=tuple(float(x) for x in ret.tolist()) if self.store_rollout_returns else None,
            features={key: float(value) for key, value in features.items()},
            metadata={"horizon": self.horizon, "bound": "empirical_bernstein"},
        )

    def _rollout_once(self, env: POMDPEnv, action: Action, rng: np.random.Generator) -> tuple[float, float]:
        sim = env.clone()
        step = sim.step(action)
        total = float(step.reward)
        immediate = float(step.reward)
        discount = self.gamma
        depth = 1
        terminated = bool(step.terminated or step.truncated or sim.is_terminal())
        while depth < self.horizon and not terminated:
            legal = list(sim.legal_actions())
            if not legal:
                break
            next_action = self.rollout_policy.select_action(sim, rng)
            if next_action not in legal:
                next_action = legal[0]
            step = sim.step(next_action)
            total += discount * float(step.reward)
            discount *= self.gamma
            depth += 1
            terminated = bool(step.terminated or step.truncated or sim.is_terminal())
        return total, immediate

    def _confidence_interval(
        self,
        returns: np.ndarray,
        env: POMDPEnv,
        n_actions: int,
    ) -> tuple[float, float]:
        r"""Empirical Bernstein confidence interval for bounded rollout returns.

        Let the discounted rollout return for an action lie in [L, U]. Setting
        B = U - L and using the empirical sample variance \hat{\sigma}^2, a standard
        empirical Bernstein radius is

            beta = \hat{\sigma} \sqrt{2 \log(3 / \delta') / n}
                   + 3 B \log(3 / \delta') / n,

        where \delta' applies a union bound over root actions. The interval is then
        [mean - beta, mean + beta], clipped to [L, U].
        """
        if returns.ndim != 1 or len(returns) == 0:
            raise ValueError("returns must be a non-empty 1D array.")

        lower, upper = env.discounted_return_bounds(self.horizon, self.gamma)
        width = float(upper - lower)
        n = int(len(returns))
        mean = float(np.mean(returns))
        if n >= 2:
            sample_var = float(np.var(returns, ddof=1))
        else:
            sample_var = 0.0
        sample_var = max(sample_var, 0.0)
        sample_std = math.sqrt(sample_var)

        # Simultaneous coverage across audited root actions.
        delta_per_action = self.delta / max(1, int(n_actions))
        log_term = math.log(3.0 / delta_per_action)
        beta = sample_std * math.sqrt(2.0 * log_term / n) + (3.0 * width * log_term / n)

        lcb = max(lower, mean - beta)
        ucb = min(upper, mean + beta)
        return float(lcb), float(ucb)

    @staticmethod
    def _cvar(values: np.ndarray, alpha: float) -> float:
        if len(values) == 0:
            return float("nan")
        k = max(1, int(math.ceil(alpha * len(values))))
        sorted_values = np.sort(values)
        return float(np.mean(sorted_values[:k]))
