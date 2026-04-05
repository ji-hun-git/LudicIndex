from __future__ import annotations

from pathlib import Path

from ludic_ai import (
    AttributeQuantilePersona,
    CallablePersona,
    DeterministicValidityGate,
    ExperimentSuite,
    FreeFormRandomSelector,
    NoPersona,
    NoisyGroundedSelector,
    RolloutOracle,
    StateBankBuilder,
    StateBankEvaluator,
)
from ludic_ai.evaluation import DecisionEvaluator
from ludic_ai.metrics import MetricEngine
from ludic_ai.persona import CompositePersona

from examples.toy_treasure_pomdp import ToyTreasurePOMDP


def build_preserving_predicate(action, audits, history):
    build = float(history.get("build", 0.0))
    if build <= 0.0:
        return True
    return action != "cash_out"


def env_factory() -> ToyTreasurePOMDP:
    return ToyTreasurePOMDP(max_turns=5, signal_accuracy=0.75, favorable_prob=0.5)


def main() -> None:
    oracle = RolloutOracle(horizon=3, n_rollouts=32, gamma=0.95, delta=0.05, top_k=3)
    gate = DeterministicValidityGate(strict_menu_membership=False)
    suite = ExperimentSuite(oracle=oracle, validator=gate)

    personas = [
        NoPersona(),
        AttributeQuantilePersona(
            name="risk_seeking",
            attribute="return_var",
            quantile=0.60,
            mode="ge",
            pressure=0.60,
        ),
        AttributeQuantilePersona(
            name="risk_seeking",
            attribute="return_var",
            quantile=0.85,
            mode="ge",
            pressure=0.85,
        ),
        AttributeQuantilePersona(
            name="loss_averse",
            attribute="cvar",
            quantile=0.60,
            mode="ge",
            pressure=0.60,
        ),
        AttributeQuantilePersona(
            name="loss_averse",
            attribute="cvar",
            quantile=0.85,
            mode="ge",
            pressure=0.85,
        ),
        CallablePersona(
            name="build_preserving",
            prompt="Avoid cashing out when build has already been invested.",
            pressure=0.80,
            predicate=build_preserving_predicate,
        ),
    ]

    selectors = [
        FreeFormRandomSelector(selector_id="free_form_random", invalid_prob=0.25, illegal_prob=0.15, capability=0.1),
        NoisyGroundedSelector(selector_id="grounded_cap_low", capability=0.30, persona_weight=2.0),
        NoisyGroundedSelector(selector_id="grounded_cap_mid", capability=0.60, persona_weight=2.0),
        NoisyGroundedSelector(selector_id="grounded_cap_high", capability=0.90, persona_weight=2.0),
    ]

    log = suite.run_episode_grid(
        env_factory=env_factory,
        selectors=selectors,
        personas=personas,
        seeds=list(range(8)),
        max_steps=5,
        experiment_id="toy_episode_grid",
    )

    outputs = Path(__file__).resolve().parent / "demo_outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    log.export(outputs)
    analyses = suite.analyze(log)
    for name, frame in analyses.items():
        frame.to_csv(outputs / f"{name}.csv", index=False)

    # Optional matched-state evaluation for high-power turn-level comparisons.
    bank_builder = StateBankBuilder(oracle=oracle)
    bank = bank_builder.collect(env_factory, seeds=list(range(4)), max_steps=5, max_items=20)
    decision_evaluator = DecisionEvaluator(
        oracle=oracle,
        validator=gate,
        metric_engine=MetricEngine(horizon=oracle.horizon, gamma=oracle.gamma),
    )
    bank_eval = StateBankEvaluator(decision_evaluator)
    bank_log = bank_eval.evaluate(
        bank=bank,
        selectors=selectors,
        personas=personas,
        experiment_id="toy_state_bank_grid",
    )
    bank_log.export(outputs / "state_bank")

    print("Wrote outputs to", outputs)
    print("Turn rows:", len(log.turn_frame()))
    print("Episode rows:", len(log.episode_frame()))
    print("Confound rate summary:")
    print(analyses["confound_rates"].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
