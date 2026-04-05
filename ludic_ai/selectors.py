from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

import numpy as np

from .persona import NoPersona, Persona
from .types import Action, OracleDecision, Selection


class BaseSelector(ABC):
    selector_id: str = "selector"
    capability: Optional[float] = None
    uses_oracle: bool = False

    @abstractmethod
    def select(
        self,
        observation: Any,
        oracle_decision: OracleDecision,
        persona: Persona,
        rng: np.random.Generator,
        history: Mapping[str, Any],
    ) -> Selection:
        raise NotImplementedError


@dataclass(slots=True)
class CallableSelector(BaseSelector):
    fn: Callable[[Any, OracleDecision, Persona, np.random.Generator, Mapping[str, Any]], Selection]
    selector_id: str = "callable_selector"
    capability: Optional[float] = None
    uses_oracle: bool = True

    def select(
        self,
        observation: Any,
        oracle_decision: OracleDecision,
        persona: Persona,
        rng: np.random.Generator,
        history: Mapping[str, Any],
    ) -> Selection:
        result = self.fn(observation, oracle_decision, persona, rng, history)
        if not result.selector_id:
            result.selector_id = self.selector_id
        return result


@dataclass(slots=True)
class OracleArgmaxSelector(BaseSelector):
    selector_id: str = "oracle_argmax"
    capability: Optional[float] = 1.0
    uses_oracle: bool = True

    def select(
        self,
        observation: Any,
        oracle_decision: OracleDecision,
        persona: Persona,
        rng: np.random.Generator,
        history: Mapping[str, Any],
    ) -> Selection:
        return Selection(
            action=oracle_decision.oracle_action_est,
            raw_output=oracle_decision.oracle_action_est,
            parsed_ok=True,
            selector_id=self.selector_id,
        )


@dataclass(slots=True)
class PersonaOracleSelector(BaseSelector):
    selector_id: str = "persona_oracle"
    capability: Optional[float] = 1.0
    uses_oracle: bool = True

    def select(
        self,
        observation: Any,
        oracle_decision: OracleDecision,
        persona: Persona,
        rng: np.random.Generator,
        history: Mapping[str, Any],
    ) -> Selection:
        action = oracle_decision.persona_action_est or oracle_decision.oracle_action_est
        return Selection(action=action, raw_output=action, parsed_ok=True, selector_id=self.selector_id)


@dataclass(slots=True)
class NoisyGroundedSelector(BaseSelector):
    selector_id: str = "noisy_grounded"
    capability: Optional[float] = 0.5
    persona_weight: float = 2.0
    temperature_floor: float = 0.02
    uses_oracle: bool = True

    def _score(
        self,
        oracle_decision: OracleDecision,
        persona: Persona,
        history: Mapping[str, Any],
    ) -> tuple[list[Action], np.ndarray]:
        menu = list(oracle_decision.menu_actions)
        means = np.array([oracle_decision.audits[action].mean for action in menu], dtype=float)
        if len(means) == 0:
            raise RuntimeError("Oracle menu is empty.")
        mean_std = float(np.std(means))
        centered = means - float(np.mean(means))
        normalized = centered / mean_std if mean_std > 1e-12 else np.zeros_like(means)
        persona_bonus = np.array(
            [1.0 if persona.is_satisfied(action, oracle_decision.audits, history) else -1.0 for action in menu],
            dtype=float,
        )
        cap = 0.5 if self.capability is None else float(self.capability)
        return menu, cap * normalized + self.persona_weight * persona_bonus

    def select(
        self,
        observation: Any,
        oracle_decision: OracleDecision,
        persona: Persona,
        rng: np.random.Generator,
        history: Mapping[str, Any],
    ) -> Selection:
        menu, base_scores = self._score(oracle_decision, persona, history)
        cap = 0.5 if self.capability is None else float(self.capability)
        temperature = max(self.temperature_floor, 1.0 - cap)
        noise = rng.gumbel(loc=0.0, scale=temperature, size=len(menu))
        chosen_idx = int(np.argmax(base_scores + noise))
        action = menu[chosen_idx]
        return Selection(
            action=action,
            raw_output={"action": action, "menu": [str(a) for a in menu]},
            parsed_ok=True,
            selector_id=self.selector_id,
            metadata={"sampling_temperature": temperature},
        )


@dataclass(slots=True)
class FreeFormRandomSelector(BaseSelector):
    selector_id: str = "free_form_random"
    capability: Optional[float] = 0.1
    invalid_prob: float = 0.2
    illegal_prob: float = 0.15
    uses_oracle: bool = False
    invalid_token: str = "<UNPARSEABLE>"
    illegal_token: str = "<ILLEGAL_ACTION>"

    def select(
        self,
        observation: Any,
        oracle_decision: OracleDecision,
        persona: Persona,
        rng: np.random.Generator,
        history: Mapping[str, Any],
    ) -> Selection:
        u = float(rng.random())
        legal = list(oracle_decision.legal_actions)
        if u < self.invalid_prob:
            return Selection(
                action=None,
                raw_output=self.invalid_token,
                parsed_ok=False,
                selector_id=self.selector_id,
            )
        if u < self.invalid_prob + self.illegal_prob:
            return Selection(
                action=self.illegal_token,
                raw_output=self.illegal_token,
                parsed_ok=True,
                selector_id=self.selector_id,
            )
        if not legal:
            return Selection(action=None, raw_output=self.invalid_token, parsed_ok=False, selector_id=self.selector_id)
        action = legal[int(rng.integers(len(legal)))]
        return Selection(action=action, raw_output=str(action), parsed_ok=True, selector_id=self.selector_id)


class AbstractLLMMenuSelector(BaseSelector, ABC):
    """Production hook for an external LLM. Subclass call_model to integrate an API client."""

    selector_id: str = "llm_menu_selector"
    capability: Optional[float] = None
    uses_oracle: bool = True

    def build_prompt(
        self,
        observation: Any,
        oracle_decision: OracleDecision,
        persona: Persona,
        history: Mapping[str, Any],
    ) -> str:
        lines = [
            f"Persona: {persona.name}",
            f"Constraint: {persona.prompt}",
            f"Observation: {observation}",
            "Audited menu:",
        ]
        for action in oracle_decision.menu_actions:
            audit = oracle_decision.audits[action]
            lines.append(
                f"- action={action!r}, mean={audit.mean:.4f}, lcb={audit.lcb:.4f}, ucb={audit.ucb:.4f}, features={audit.features}"
            )
        lines.append("Return a JSON object with a single field 'action'.")
        return "\n".join(lines)

    @abstractmethod
    def call_model(self, prompt: str) -> Any:
        raise NotImplementedError

    def parse_model_output(self, raw_output: Any, oracle_decision: OracleDecision) -> Selection:
        if isinstance(raw_output, dict) and "action" in raw_output:
            action = raw_output["action"]
            return Selection(
                action=action,
                raw_output=raw_output,
                parsed_ok=True,
                selector_id=self.selector_id,
            )
        return Selection(action=None, raw_output=raw_output, parsed_ok=False, selector_id=self.selector_id)

    def select(
        self,
        observation: Any,
        oracle_decision: OracleDecision,
        persona: Persona,
        rng: np.random.Generator,
        history: Mapping[str, Any],
    ) -> Selection:
        prompt = self.build_prompt(observation, oracle_decision, persona, history)
        raw_output = self.call_model(prompt)
        result = self.parse_model_output(raw_output, oracle_decision)
        result.metadata["prompt"] = prompt
        return result
