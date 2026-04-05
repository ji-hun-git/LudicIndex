from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .persona import Persona
from .types import Action, OracleDecision, Selection, ValidityResult


@dataclass(slots=True)
class DeterministicValidityGate:
    strict_menu_membership: bool = False

    def validate(
        self,
        selection: Selection,
        oracle_decision: OracleDecision,
        persona: Persona,
        history: Mapping[str, Any],
    ) -> ValidityResult:
        reasons: list[str] = []
        parse_ok = bool(selection.parsed_ok and selection.action is not None)
        if not parse_ok:
            reasons.append("parse_failed")
        menu_ok = parse_ok and selection.action in set(oracle_decision.menu_actions)
        if self.strict_menu_membership and not menu_ok:
            reasons.append("not_in_menu")
        legal_ok = parse_ok and selection.action in set(oracle_decision.legal_actions)
        if parse_ok and not legal_ok:
            reasons.append("illegal_action")
        persona_ok = legal_ok and persona.is_satisfied(selection.action, oracle_decision.audits, history)
        if legal_ok and not persona_ok:
            reasons.append("persona_violation")
        overall = parse_ok and legal_ok and persona_ok and (menu_ok or not self.strict_menu_membership)
        return ValidityResult(
            parse_ok=parse_ok,
            menu_ok=menu_ok,
            legal_ok=legal_ok,
            persona_ok=persona_ok,
            overall=overall,
            reasons=tuple(reasons),
        )
