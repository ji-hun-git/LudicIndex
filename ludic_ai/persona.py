from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .types import Action, ActionAudit


class Persona(ABC):
    """Deterministic behavioral specification over audited action attributes."""

    name: str = "persona"
    prompt: str = ""
    pressure: float = 0.0

    @abstractmethod
    def is_satisfied(
        self,
        action: Action,
        audits: Mapping[Action, ActionAudit],
        history: Mapping[str, Any],
    ) -> bool:
        raise NotImplementedError

    def feasible_actions(
        self,
        audits: Mapping[Action, ActionAudit],
        history: Mapping[str, Any],
    ) -> list[Action]:
        return [action for action in audits if self.is_satisfied(action, audits, history)]

    def metadata(self) -> dict[str, Any]:
        return {
            "persona_name": self.name,
            "persona_prompt": self.prompt,
            "persona_pressure": self.pressure,
        }


@dataclass(slots=True)
class NoPersona(Persona):
    name: str = "neutral"
    prompt: str = "Maximize audited expected utility."
    pressure: float = 0.0

    def is_satisfied(
        self,
        action: Action,
        audits: Mapping[Action, ActionAudit],
        history: Mapping[str, Any],
    ) -> bool:
        return True


@dataclass(slots=True)
class AttributeQuantilePersona(Persona):
    attribute: str
    quantile: float
    mode: str = "ge"
    name: str = "attribute_quantile"
    prompt: str = ""
    pressure: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.quantile <= 1.0:
            raise ValueError("quantile must be in [0, 1].")
        if self.mode not in {"ge", "le"}:
            raise ValueError("mode must be one of {'ge', 'le'}.")
        if not self.prompt:
            comparator = "top" if self.mode == "ge" else "bottom"
            self.prompt = (
                f"Prefer actions in the {comparator} {self.quantile:.2f} quantile of {self.attribute}."
            )
        if self.pressure == 0.0:
            self.pressure = self.quantile

    def _threshold(self, audits: Mapping[Action, ActionAudit]) -> float:
        values = [audit.features[self.attribute] for audit in audits.values() if self.attribute in audit.features]
        if not values:
            raise KeyError(f"Attribute '{self.attribute}' not found in audited features.")
        return float(np.quantile(values, self.quantile))

    def is_satisfied(
        self,
        action: Action,
        audits: Mapping[Action, ActionAudit],
        history: Mapping[str, Any],
    ) -> bool:
        audit = audits[action]
        if self.attribute not in audit.features:
            raise KeyError(f"Attribute '{self.attribute}' not found for action {action!r}.")
        threshold = self._threshold(audits)
        value = audit.features[self.attribute]
        return bool(value >= threshold) if self.mode == "ge" else bool(value <= threshold)


@dataclass(slots=True)
class FeatureThresholdPersona(Persona):
    attribute: str
    threshold: float
    mode: str = "ge"
    name: str = "feature_threshold"
    prompt: str = ""
    pressure: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in {"ge", "le"}:
            raise ValueError("mode must be one of {'ge', 'le'}.")
        if not self.prompt:
            symbol = ">=" if self.mode == "ge" else "<="
            self.prompt = f"Only choose actions with {self.attribute} {symbol} {self.threshold}."
        if self.pressure == 0.0:
            self.pressure = abs(self.threshold)

    def is_satisfied(
        self,
        action: Action,
        audits: Mapping[Action, ActionAudit],
        history: Mapping[str, Any],
    ) -> bool:
        audit = audits[action]
        if self.attribute not in audit.features:
            raise KeyError(f"Attribute '{self.attribute}' not found for action {action!r}.")
        value = audit.features[self.attribute]
        return bool(value >= self.threshold) if self.mode == "ge" else bool(value <= self.threshold)


@dataclass(slots=True)
class CallablePersona(Persona):
    predicate: Callable[[Action, Mapping[Action, ActionAudit], Mapping[str, Any]], bool]
    name: str = "callable_persona"
    prompt: str = "Apply deterministic constraint."
    pressure: float = 0.0
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    def is_satisfied(
        self,
        action: Action,
        audits: Mapping[Action, ActionAudit],
        history: Mapping[str, Any],
    ) -> bool:
        return bool(self.predicate(action, audits, history))

    def metadata(self) -> dict[str, Any]:
        payload = super().metadata()
        payload.update(self.extra_metadata)
        return payload


@dataclass(slots=True)
class CompositePersona(Persona):
    components: Sequence[Persona]
    name: str = "composite_persona"
    prompt: str = ""
    pressure: float = 0.0

    def __post_init__(self) -> None:
        if not self.prompt:
            self.prompt = " AND ".join(component.prompt for component in self.components)
        if self.pressure == 0.0:
            self.pressure = float(sum(component.pressure for component in self.components))

    def is_satisfied(
        self,
        action: Action,
        audits: Mapping[Action, ActionAudit],
        history: Mapping[str, Any],
    ) -> bool:
        return all(component.is_satisfied(action, audits, history) for component in self.components)
