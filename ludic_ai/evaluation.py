from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np
import pandas as pd

from .env import POMDPEnv
from .metrics import MetricEngine
from .oracle import RolloutOracle
from .persona import NoPersona, Persona
from .records import TurnLog
from .selectors import BaseSelector
from .stats import ExperimentAnalyzer
from .types import OracleDecision, TurnRecord
from .validator import DeterministicValidityGate


@dataclass(slots=True)
class DecisionEvaluator:
    oracle: RolloutOracle
    validator: DeterministicValidityGate
    metric_engine: MetricEngine
    invalid_action_terminates: bool = True

    @staticmethod
    def _phase_rngs(
        *,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[np.random.Generator, np.random.Generator, np.random.Generator]:
        """Create independent RNG streams for audit and selection.

        The master RNG is advanced only to generate child seeds; thereafter the oracle and the
        selector consume disjoint generators. This prevents selector behavior from depending on
        oracle rollout count, action ordering, or audit horizon.
        """
        master_rng = rng if rng is not None else np.random.default_rng(seed)
        child_seeds = master_rng.integers(0, np.iinfo(np.uint64).max, size=2, dtype=np.uint64)
        oracle_rng = np.random.default_rng(int(child_seeds[0]))
        selector_rng = np.random.default_rng(int(child_seeds[1]))
        return master_rng, oracle_rng, selector_rng

    def evaluate_state(
        self,
        env: POMDPEnv,
        selector: BaseSelector,
        persona: Optional[Persona] = None,
        *,
        experiment_id: str,
        run_id: str,
        episode_id: str,
        step_index: int,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        precomputed_decision: Optional[OracleDecision] = None,
        execute_action: bool = True,
    ) -> TurnRecord:
        persona = persona or NoPersona()
        _, oracle_rng, selector_rng = self._phase_rngs(seed=seed, rng=rng)
        history = dict(env.history_context())
        if precomputed_decision is None:
            oracle_decision = self.oracle.audit(env, persona=persona, history=history, rng=oracle_rng)
        else:
            oracle_decision = self.oracle.rebuild_decision(
                env,
                precomputed_decision.audits,
                persona=persona,
                history=history,
            )
        selection = selector.select(env.observe(), oracle_decision, persona, selector_rng, history)
        validity = self.validator.validate(selection, oracle_decision, persona, history)
        metrics, metric_meta = self.metric_engine.compute(env, oracle_decision, selection, validity, persona, history)

        reward = math.nan
        terminated = False
        invalid_termination = False
        if execute_action:
            if selection.action in set(oracle_decision.legal_actions):
                step = env.step(selection.action)
                reward = float(step.reward)
                terminated = bool(step.terminated or step.truncated or env.is_terminal())
            else:
                reward = float(env.invalid_action_reward(self.oracle.horizon, self.oracle.gamma))
                terminated = self.invalid_action_terminates
                invalid_termination = True

        metadata = {
            "raw_output": repr(selection.raw_output),
            "validity_reasons": "|".join(validity.reasons),
            **metric_meta,
            **{f"selector__{k}": v for k, v in selection.metadata.items()},
            **{f"oracle__{k}": v for k, v in oracle_decision.metadata.items()},
        }
        return TurnRecord(
            experiment_id=experiment_id,
            run_id=run_id,
            episode_id=episode_id,
            step_index=step_index,
            seed=seed,
            selector_id=selector.selector_id,
            selector_capability=selector.capability,
            selector_uses_oracle=selector.uses_oracle,
            persona_name=persona.name,
            persona_pressure=persona.pressure,
            state_id=oracle_decision.state_id,
            selected_action=selection.action,
            oracle_action_est=oracle_decision.oracle_action_est,
            oracle_action_true=oracle_decision.oracle_action_true,
            persona_action_est=oracle_decision.persona_action_est,
            persona_action_true=oracle_decision.persona_action_true,
            parse_ok=validity.parse_ok,
            menu_ok=validity.menu_ok,
            legal_ok=validity.legal_ok,
            persona_ok=validity.persona_ok,
            A_t=validity.overall,
            gap_est=metrics.gap_est,
            li=metrics.li,
            vli_op=metrics.vli_op,
            vli_cert=metrics.vli_cert,
            cc_est=metrics.cc_est,
            rpr_est=metrics.rpr_est,
            cc_cert=metrics.cc_cert,
            rpr_cert=metrics.rpr_cert,
            pressure_est=metrics.pressure_est,
            gap_true=metrics.gap_true,
            cc_true=metrics.cc_true,
            rpr_true=metrics.rpr_true,
            pressure_true=metrics.pressure_true,
            oracle_copy_est=metrics.oracle_copy_est,
            persona_oracle_copy_est=metrics.persona_oracle_copy_est,
            oracle_copy_true=metrics.oracle_copy_true,
            persona_oracle_copy_true=metrics.persona_oracle_copy_true,
            confounded_positive_gap=metrics.confounded_positive_gap,
            selected_proxy_value_used=metrics.selected_proxy_value_used,
            reward=reward,
            terminated=terminated,
            invalid_termination=invalid_termination,
            menu_size=len(oracle_decision.menu_actions),
            legal_action_count=len(oracle_decision.legal_actions),
            metadata=metadata,
        )


@dataclass(slots=True)
class EpisodeRunner:
    decision_evaluator: DecisionEvaluator

    def run_episode(
        self,
        env_factory: Callable[[], POMDPEnv],
        selector: BaseSelector,
        persona: Optional[Persona] = None,
        *,
        seed: Optional[int] = None,
        max_steps: int = 100,
        experiment_id: str = "episode_grid",
        episode_id: Optional[str] = None,
    ) -> list[TurnRecord]:
        env = env_factory()
        env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        persona = persona or NoPersona()
        episode_id = episode_id or (f"seed_{seed}" if seed is not None else "episode")
        run_id = f"{selector.selector_id}|{persona.name}|seed={seed}"
        records: list[TurnRecord] = []
        for step_index in range(max_steps):
            if env.is_terminal():
                break
            record = self.decision_evaluator.evaluate_state(
                env,
                selector,
                persona,
                experiment_id=experiment_id,
                run_id=run_id,
                episode_id=episode_id,
                step_index=step_index,
                seed=seed,
                rng=rng,
                precomputed_decision=None,
                execute_action=True,
            )
            records.append(record)
            if record.terminated:
                break
        return records


@dataclass(slots=True)
class ExperimentSuite:
    oracle: RolloutOracle
    validator: DeterministicValidityGate
    analyzer: ExperimentAnalyzer = field(default_factory=ExperimentAnalyzer)
    invalid_action_terminates: bool = True

    def _decision_evaluator(self) -> DecisionEvaluator:
        return DecisionEvaluator(
            oracle=self.oracle,
            validator=self.validator,
            metric_engine=MetricEngine(horizon=self.oracle.horizon, gamma=self.oracle.gamma),
            invalid_action_terminates=self.invalid_action_terminates,
        )

    def run_episode_grid(
        self,
        env_factory: Callable[[], POMDPEnv],
        selectors: Sequence[BaseSelector],
        personas: Sequence[Persona],
        seeds: Sequence[int],
        *,
        max_steps: int = 100,
        experiment_id: str = "episode_grid",
    ) -> TurnLog:
        log = TurnLog()
        runner = EpisodeRunner(self._decision_evaluator())
        for selector in selectors:
            for persona in personas:
                for seed in seeds:
                    records = runner.run_episode(
                        env_factory,
                        selector,
                        persona,
                        seed=seed,
                        max_steps=max_steps,
                        experiment_id=experiment_id,
                        episode_id=f"seed_{seed}",
                    )
                    log.extend(records)
        return log

    def analyze(self, log: TurnLog) -> dict[str, pd.DataFrame]:
        turns = log.turn_frame()
        episodes = log.episode_frame()
        return self.analyzer.analyze(turns, episodes)
