from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np

from ludic_ai.env import POMDPEnv, RolloutPolicy
from ludic_ai.types import Action, StepResult


ACTIONS = ("safe", "risky", "setup", "cash_out")


def build_preserving_predicate(action, audits, history):
    """Deterministic predicate used by the reference build-preserving persona.

    The persona remains neutral until a positive build has been accumulated. Once build exists,
    the predicate forbids immediate liquidation via ``cash_out`` and therefore encodes a simple
    commitment-preserving endogenous constraint.
    """
    build = float(history.get("build", 0.0))
    if build <= 0.0:
        return True
    return action != "cash_out"



def build_env(**kwargs) -> "ToyTreasurePOMDP":
    """Factory helper used by the config-driven runner.

    Example config path:
        environment:
          factory: examples.toy_treasure_pomdp:build_env
          kwargs:
            max_turns: 5
            signal_accuracy: 0.75
            favorable_prob: 0.5
    """
    return ToyTreasurePOMDP(**kwargs)


@dataclass
class ToyTreasurePOMDP(POMDPEnv):
    """A small stochastic, partially observed environment with tunable constraint trade-offs.

    State:
      - turn index
      - latent boolean ``favorable``
      - observable noisy signal of ``favorable``
      - build counter accumulated by repeated ``setup``

    Actions:
      - safe: deterministic moderate reward
      - risky: high variance reward conditioned on latent state
      - setup: small immediate reward, increases build
      - cash_out: realizes build-dependent reward and resets build
    """

    max_turns: int = 5
    signal_accuracy: float = 0.75
    favorable_prob: float = 0.5
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    turn: int = 0
    build: int = 0
    favorable: bool = False
    terminal: bool = False
    last_signal: str = "unknown"

    def reset(self, seed: Optional[int] = None) -> Any:
        self.rng = np.random.default_rng(seed)
        self.turn = 0
        self.build = 0
        self.terminal = False
        self.favorable = bool(self.rng.random() < self.favorable_prob)
        self.last_signal = self._emit_signal(self.favorable)
        return self.observe()

    def clone(self) -> "ToyTreasurePOMDP":
        clone = ToyTreasurePOMDP(
            max_turns=self.max_turns,
            signal_accuracy=self.signal_accuracy,
            favorable_prob=self.favorable_prob,
        )
        clone.turn = self.turn
        clone.build = self.build
        clone.favorable = self.favorable
        clone.terminal = self.terminal
        clone.last_signal = self.last_signal
        clone.rng = np.random.default_rng()
        clone.rng.bit_generator.state = self.rng.bit_generator.state
        return clone

    def observe(self) -> Any:
        return {
            "turn": self.turn,
            "build": self.build,
            "signal": self.last_signal,
            "turns_remaining": self.max_turns - self.turn,
        }

    def legal_actions(self):
        return ACTIONS

    def step(self, action: Action) -> StepResult:
        if self.terminal:
            raise RuntimeError("Cannot step a terminated environment.")
        if action not in ACTIONS:
            raise ValueError(f"Illegal action: {action!r}")
        reward = self._reward(action, favorable=self.favorable, build=self.build)
        self.build = self._next_build(action, self.build)
        self.turn += 1
        self.terminal = self.turn >= self.max_turns
        if not self.terminal:
            self.favorable = bool(self.rng.random() < self.favorable_prob)
            self.last_signal = self._emit_signal(self.favorable)
        return StepResult(
            observation=self.observe(),
            reward=float(reward),
            terminated=self.terminal,
            truncated=False,
            info={"action": action, "favorable": self.favorable, "build": self.build},
        )

    def is_terminal(self) -> bool:
        return self.terminal

    def state_id(self) -> str:
        return f"t={self.turn}|build={self.build}|fav={int(self.favorable)}|signal={self.last_signal}"

    def reward_range(self) -> tuple[float, float]:
        upper_cash = 2.0 + 5.0 * self.max_turns
        return -5.0, max(12.0, upper_cash)

    def history_context(self) -> Mapping[str, Any]:
        return {"turn": self.turn, "build": self.build, "signal": self.last_signal}

    def extra_action_features(self, action: Action, rollout_returns, immediate_rewards):
        current_build = float(self.build)
        features = {
            "current_build": current_build,
            "preserves_build": 1.0 if current_build > 0 and action != "cash_out" else 0.0,
            "realizes_build": 1.0 if action == "cash_out" and current_build > 0 else 0.0,
            "risk_bonus": 1.0 if action == "risky" else 0.0,
            "immediacy_bonus": 1.0 if action in {"safe", "cash_out"} else 0.0,
        }
        return features

    def exact_action_value(
        self,
        action: Action,
        horizon: int,
        gamma: float,
        rollout_policy: RolloutPolicy,
    ) -> Optional[float]:
        temp = self.clone()
        return self._exact_from_state(
            turn=temp.turn,
            build=temp.build,
            favorable=temp.favorable,
            action=action,
            horizon=horizon,
            gamma=gamma,
            rollout_policy=rollout_policy,
        )

    def _emit_signal(self, favorable: bool) -> str:
        honest = bool(self.rng.random() < self.signal_accuracy)
        if honest:
            return "good" if favorable else "bad"
        return "bad" if favorable else "good"

    @staticmethod
    def _reward(action: Action, favorable: bool, build: int) -> float:
        if action == "safe":
            return 4.0
        if action == "risky":
            return 12.0 if favorable else -5.0
        if action == "setup":
            return 1.0
        if action == "cash_out":
            return 2.0 + 5.0 * build
        raise ValueError(action)

    @staticmethod
    def _next_build(action: Action, build: int) -> int:
        if action == "setup":
            return build + 1
        if action == "cash_out":
            return 0
        return build

    def _exact_from_state(
        self,
        turn: int,
        build: int,
        favorable: bool,
        action: Action,
        horizon: int,
        gamma: float,
        rollout_policy: RolloutPolicy,
    ) -> float:
        reward = self._reward(action, favorable=favorable, build=build)
        next_turn = turn + 1
        next_build = self._next_build(action, build)
        if horizon <= 1 or next_turn >= self.max_turns:
            return float(reward)
        future_expectation = 0.0
        for next_favorable, p_fav in [(False, 1.0 - self.favorable_prob), (True, self.favorable_prob)]:
            next_env = ToyTreasurePOMDP(
                max_turns=self.max_turns,
                signal_accuracy=self.signal_accuracy,
                favorable_prob=self.favorable_prob,
            )
            next_env.turn = next_turn
            next_env.build = next_build
            next_env.favorable = next_favorable
            next_env.terminal = next_turn >= self.max_turns
            next_env.last_signal = "good" if next_favorable else "bad"
            action_dist = rollout_policy.action_distribution(next_env)
            if action_dist is None:
                return float("nan")
            subtotal = 0.0
            for next_action, p_action in action_dist.items():
                subtotal += float(p_action) * self._exact_from_state(
                    turn=next_turn,
                    build=next_build,
                    favorable=next_favorable,
                    action=next_action,
                    horizon=horizon - 1,
                    gamma=gamma,
                    rollout_policy=rollout_policy,
                )
            future_expectation += float(p_fav) * subtotal
        return float(reward + gamma * future_expectation)
