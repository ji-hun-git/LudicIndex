from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .types import Action, StepResult


class RolloutPolicy(ABC):
    """Policy used for continuation rollouts inside the oracle."""

    name: str = "rollout_policy"

    @abstractmethod
    def select_action(self, env: "POMDPEnv", rng: np.random.Generator) -> Action:
        raise NotImplementedError

    def action_distribution(self, env: "POMDPEnv") -> Optional[Mapping[Action, float]]:
        return None


class RandomRolloutPolicy(RolloutPolicy):
    name = "uniform_random"

    def select_action(self, env: "POMDPEnv", rng: np.random.Generator) -> Action:
        legal = list(env.legal_actions())
        if not legal:
            raise RuntimeError("No legal actions available for random rollout policy.")
        return legal[int(rng.integers(len(legal)))]

    def action_distribution(self, env: "POMDPEnv") -> Optional[Mapping[Action, float]]:
        legal = list(env.legal_actions())
        if not legal:
            return {}
        p = 1.0 / len(legal)
        return {action: p for action in legal}


class POMDPEnv(ABC):
    """Abstract environment interface for local audited evaluation."""

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Any:
        raise NotImplementedError

    @abstractmethod
    def clone(self) -> "POMDPEnv":
        raise NotImplementedError

    @abstractmethod
    def observe(self) -> Any:
        raise NotImplementedError

    def oracle_view(self) -> Any:
        return self.observe()

    @abstractmethod
    def legal_actions(self) -> Sequence[Action]:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: Action) -> StepResult:
        raise NotImplementedError

    @abstractmethod
    def is_terminal(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def state_id(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def reward_range(self) -> tuple[float, float]:
        """Return a per-step bounded reward interval used for concentration bounds."""
        raise NotImplementedError

    def history_context(self) -> Mapping[str, Any]:
        return {}

    def extra_action_features(
        self,
        action: Action,
        rollout_returns: Sequence[float],
        immediate_rewards: Sequence[float],
    ) -> Mapping[str, float]:
        return {}

    def exact_action_value(
        self,
        action: Action,
        horizon: int,
        gamma: float,
        rollout_policy: RolloutPolicy,
    ) -> Optional[float]:
        return None

    def discounted_return_bounds(self, horizon: int, gamma: float) -> tuple[float, float]:
        r_min, r_max = self.reward_range()
        if horizon <= 0:
            return 0.0, 0.0
        if gamma == 1.0:
            factor = float(horizon)
        else:
            factor = float((1.0 - gamma**horizon) / (1.0 - gamma))
        return r_min * factor, r_max * factor

    def invalid_action_reward(self, horizon: int, gamma: float) -> float:
        lower, _ = self.discounted_return_bounds(horizon=horizon, gamma=gamma)
        return lower

    def deepcopy(self) -> "POMDPEnv":
        return copy.deepcopy(self)
