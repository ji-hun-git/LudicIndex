from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXAMPLES))

from ludic_ai import (  # noqa: E402
    AttributeQuantilePersona,
    DeterministicValidityGate,
    ExperimentSuite,
    FreeFormRandomSelector,
    NoPersona,
    NoisyGroundedSelector,
    RolloutOracle,
)
from toy_treasure_pomdp import ToyTreasurePOMDP  # noqa: E402


def env_factory():
    return ToyTreasurePOMDP(max_turns=4)


def run_smoke() -> None:
    oracle = RolloutOracle(horizon=3, n_rollouts=32, gamma=0.95, top_k=3)
    suite = ExperimentSuite(oracle=oracle, validator=DeterministicValidityGate())
    selectors = [
        FreeFormRandomSelector(selector_id="ff", invalid_prob=0.2, illegal_prob=0.1),
        NoisyGroundedSelector(selector_id="grounded", capability=0.8),
    ]
    personas = [
        NoPersona(),
        AttributeQuantilePersona(name="risk", attribute="return_var", quantile=0.7, mode="ge", pressure=0.7),
    ]
    log = suite.run_episode_grid(env_factory, selectors, personas, seeds=[0, 1, 2], max_steps=4)
    turns = log.turn_frame()
    episodes = log.episode_frame()
    analyses = suite.analyze(log)

    assert not turns.empty
    assert not episodes.empty
    required_cols = {"vli_cert", "cc_true", "rpr_true", "confounded_positive_gap", "A_t"}
    assert required_cols.issubset(turns.columns)
    assert "sum_vli_cert" in episodes.columns
    assert "confound_rates" in analyses


if __name__ == "__main__":
    run_smoke()
    print("smoke test passed")
