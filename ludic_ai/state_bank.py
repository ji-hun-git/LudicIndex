from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from .env import POMDPEnv
from .evaluation import DecisionEvaluator
from .oracle import RolloutOracle
from .persona import NoPersona, Persona
from .records import TurnLog
from .selectors import BaseSelector, FreeFormRandomSelector
from .types import OracleDecision


@dataclass(slots=True)
class StateBankItem:
    item_id: str
    env_snapshot: POMDPEnv
    oracle_decision: OracleDecision
    episode_id: str
    step_index: int
    seed: Optional[int]
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class StateBank:
    items: list[StateBankItem] = field(default_factory=list)

    def add(self, item: StateBankItem) -> None:
        self.items.append(item)

    def __len__(self) -> int:
        return len(self.items)


@dataclass(slots=True)
class StateBankBuilder:
    oracle: RolloutOracle
    exploration_selector: BaseSelector = field(
        default_factory=lambda: FreeFormRandomSelector(
            selector_id="bank_explorer",
            invalid_prob=0.0,
            illegal_prob=0.0,
            capability=0.0,
        )
    )

    @staticmethod
    def _phase_rngs(master_rng: np.random.Generator) -> tuple[np.random.Generator, np.random.Generator]:
        child_seeds = master_rng.integers(0, np.iinfo(np.uint64).max, size=2, dtype=np.uint64)
        oracle_rng = np.random.default_rng(int(child_seeds[0]))
        selector_rng = np.random.default_rng(int(child_seeds[1]))
        return oracle_rng, selector_rng

    def collect(
        self,
        env_factory: Callable[[], POMDPEnv],
        seeds: Sequence[int],
        *,
        max_steps: int = 100,
        max_items: Optional[int] = None,
    ) -> StateBank:
        bank = StateBank()
        neutral = NoPersona()
        for seed in seeds:
            env = env_factory()
            env.reset(seed=seed)
            master_rng = np.random.default_rng(seed)
            episode_id = f"seed_{seed}"
            for step_index in range(max_steps):
                if env.is_terminal():
                    break
                oracle_rng, selector_rng = self._phase_rngs(master_rng)
                oracle_decision = self.oracle.audit(
                    env,
                    persona=neutral,
                    history=env.history_context(),
                    rng=oracle_rng,
                )
                bank.add(
                    StateBankItem(
                        item_id=f"{episode_id}_t{step_index}",
                        env_snapshot=env.clone(),
                        oracle_decision=oracle_decision,
                        episode_id=episode_id,
                        step_index=step_index,
                        seed=seed,
                    )
                )
                if max_items is not None and len(bank) >= max_items:
                    return bank
                selection = self.exploration_selector.select(
                    env.observe(),
                    oracle_decision,
                    neutral,
                    selector_rng,
                    env.history_context(),
                )
                if selection.action in set(env.legal_actions()):
                    step = env.step(selection.action)
                    if step.terminated or step.truncated or env.is_terminal():
                        break
                else:
                    break
        return bank


@dataclass(slots=True)
class StateBankEvaluator:
    decision_evaluator: DecisionEvaluator

    def evaluate(
        self,
        bank: StateBank,
        selectors: Sequence[BaseSelector],
        personas: Sequence[Persona],
        *,
        experiment_id: str = "state_bank_grid",
    ) -> TurnLog:
        log = TurnLog()
        for selector in selectors:
            for persona in personas:
                for item in bank.items:
                    env = item.env_snapshot.clone()
                    run_id = f"{selector.selector_id}|{persona.name}|bank"
                    record = self.decision_evaluator.evaluate_state(
                        env,
                        selector,
                        persona,
                        experiment_id=experiment_id,
                        run_id=run_id,
                        episode_id=item.episode_id,
                        step_index=item.step_index,
                        seed=item.seed,
                        precomputed_decision=item.oracle_decision,
                        execute_action=False,
                    )
                    log.add(record)
        return log
