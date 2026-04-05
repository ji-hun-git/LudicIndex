from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from ludic_ai import (
    AttributeQuantilePersona,
    CallablePersona,
    CompositePersona,
    DecisionEvaluator,
    DeterministicValidityGate,
    ExperimentSuite,
    FeatureThresholdPersona,
    FreeFormRandomSelector,
    MetricEngine,
    NoPersona,
    NoisyGroundedSelector,
    OracleArgmaxSelector,
    PersonaOracleSelector,
    RolloutOracle,
    StateBankBuilder,
    StateBankEvaluator,
)
from ludic_ai.env import RandomRolloutPolicy


PERSONA_REGISTRY = {
    "NoPersona": NoPersona,
    "neutral": NoPersona,
    "AttributeQuantilePersona": AttributeQuantilePersona,
    "attribute_quantile": AttributeQuantilePersona,
    "FeatureThresholdPersona": FeatureThresholdPersona,
    "feature_threshold": FeatureThresholdPersona,
    "CallablePersona": CallablePersona,
    "callable": CallablePersona,
    "CompositePersona": CompositePersona,
    "composite": CompositePersona,
}

SELECTOR_REGISTRY = {
    "FreeFormRandomSelector": FreeFormRandomSelector,
    "free_form_random": FreeFormRandomSelector,
    "NoisyGroundedSelector": NoisyGroundedSelector,
    "noisy_grounded": NoisyGroundedSelector,
    "OracleArgmaxSelector": OracleArgmaxSelector,
    "oracle_argmax": OracleArgmaxSelector,
    "PersonaOracleSelector": PersonaOracleSelector,
    "persona_oracle": PersonaOracleSelector,
}



def load_dotted_object(spec: str) -> Any:
    if ":" not in spec:
        raise ValueError(f"Expected dotted path in module:object format, got: {spec!r}")
    module_name, object_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, object_name)



def instantiate_rollout_policy(config: Mapping[str, Any] | None):
    if not config:
        return RandomRolloutPolicy()
    policy_type = config.get("type") or config.get("class") or "ludic_ai.env:RandomRolloutPolicy"
    cls = load_dotted_object(policy_type) if ":" in str(policy_type) else None
    if cls is None:
        raise ValueError(f"Unsupported rollout policy spec: {policy_type!r}")
    kwargs = dict(config.get("kwargs", {}))
    return cls(**kwargs)



def instantiate_persona(config: Mapping[str, Any]):
    config = dict(config)
    persona_type = config.pop("type", None) or config.get("name", None) or "NoPersona"
    if isinstance(persona_type, str) and ":" in persona_type:
        cls = load_dotted_object(persona_type)
    else:
        cls = PERSONA_REGISTRY.get(str(persona_type))
    if cls is None:
        raise ValueError(f"Unknown persona type: {persona_type!r}")

    if cls is NoPersona:
        return NoPersona()

    if cls is CallablePersona:
        predicate_spec = config.pop("predicate")
        predicate = load_dotted_object(predicate_spec) if isinstance(predicate_spec, str) else predicate_spec
        return CallablePersona(predicate=predicate, **config)

    if cls is CompositePersona:
        components = [instantiate_persona(item) for item in config.pop("components")]
        return CompositePersona(components=components, **config)

    return cls(**config)



def instantiate_selector(config: Mapping[str, Any]):
    config = dict(config)
    selector_type = config.pop("type", None)
    if selector_type is None:
        raise ValueError("Each selector config must declare a 'type'.")
    if isinstance(selector_type, str) and ":" in selector_type:
        cls = load_dotted_object(selector_type)
    else:
        cls = SELECTOR_REGISTRY.get(str(selector_type))
    if cls is None:
        raise ValueError(f"Unknown selector type: {selector_type!r}")
    return cls(**config)



def build_env_factory(config: Mapping[str, Any]) -> Callable[[], Any]:
    factory_spec = config["factory"]
    factory = load_dotted_object(factory_spec)
    kwargs = dict(config.get("kwargs", {}))

    def _factory():
        return factory(**kwargs)

    return _factory



def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Ludic AI pipeline from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to the YAML experiment config.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    env_factory = build_env_factory(cfg["environment"])

    oracle_cfg = dict(cfg["oracle"])
    rollout_policy = instantiate_rollout_policy(oracle_cfg.pop("rollout_policy", None))
    oracle = RolloutOracle(rollout_policy=rollout_policy, **oracle_cfg)

    gate_cfg = dict(cfg.get("validator", {}))
    gate = DeterministicValidityGate(**gate_cfg)
    suite = ExperimentSuite(oracle=oracle, validator=gate)

    selectors = [instantiate_selector(item) for item in cfg["selectors"]]
    personas = [instantiate_persona(item) for item in cfg["personas"]]

    eval_cfg = dict(cfg.get("evaluation", {}))
    seeds = list(eval_cfg.get("seeds", []))
    if not seeds:
        raise ValueError("evaluation.seeds must be a non-empty list.")
    max_steps = int(eval_cfg.get("max_steps", 100))
    experiment_id = str(cfg.get("experiment_id", Path(args.config).stem))
    output_dir = Path(cfg.get("output_dir", f"results/{experiment_id}"))
    output_dir.mkdir(parents=True, exist_ok=True)

    log = suite.run_episode_grid(
        env_factory=env_factory,
        selectors=selectors,
        personas=personas,
        seeds=seeds,
        max_steps=max_steps,
        experiment_id=experiment_id,
    )
    log.export(output_dir)
    analyses = suite.analyze(log)
    for name, frame in analyses.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)

    if eval_cfg.get("build_state_bank", True):
        bank_builder = StateBankBuilder(oracle=oracle)
        bank = bank_builder.collect(
            env_factory,
            seeds=list(eval_cfg.get("state_bank_seeds", seeds)),
            max_steps=int(eval_cfg.get("state_bank_max_steps", max_steps)),
            max_items=eval_cfg.get("state_bank_max_items"),
        )
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
            experiment_id=f"{experiment_id}_state_bank",
        )
        state_bank_dir = output_dir / "state_bank"
        state_bank_dir.mkdir(parents=True, exist_ok=True)
        bank_log.export(state_bank_dir)
        analyses_bank = suite.analyze(bank_log)
        for name, frame in analyses_bank.items():
            frame.to_csv(state_bank_dir / f"{name}.csv", index=False)

    print(f"[OK] wrote outputs to {output_dir}")
    print(f"[OK] turn rows: {len(log.turn_frame())}")
    print(f"[OK] episode rows: {len(log.episode_frame())}")


if __name__ == "__main__":
    main()
