"""Ludic AI: competence-controlled evaluation for persona-conditioned suboptimization."""

from .env import POMDPEnv, RolloutPolicy, RandomRolloutPolicy
from .evaluation import DecisionEvaluator, EpisodeRunner, ExperimentSuite
from .metrics import MetricEngine
from .oracle import RolloutOracle
from .persona import (
    AttributeQuantilePersona,
    CallablePersona,
    CompositePersona,
    FeatureThresholdPersona,
    NoPersona,
    Persona,
)
from .records import TurnLog
from .selectors import (
    AbstractLLMMenuSelector,
    BaseSelector,
    CallableSelector,
    FreeFormRandomSelector,
    NoisyGroundedSelector,
    OracleArgmaxSelector,
    PersonaOracleSelector,
)
from .state_bank import StateBank, StateBankBuilder, StateBankEvaluator
from .stats import ExperimentAnalyzer
from .validator import DeterministicValidityGate

__all__ = [
    "POMDPEnv",
    "RolloutPolicy",
    "RandomRolloutPolicy",
    "DecisionEvaluator",
    "EpisodeRunner",
    "ExperimentSuite",
    "MetricEngine",
    "RolloutOracle",
    "AttributeQuantilePersona",
    "CallablePersona",
    "CompositePersona",
    "FeatureThresholdPersona",
    "NoPersona",
    "Persona",
    "TurnLog",
    "AbstractLLMMenuSelector",
    "BaseSelector",
    "CallableSelector",
    "FreeFormRandomSelector",
    "NoisyGroundedSelector",
    "OracleArgmaxSelector",
    "PersonaOracleSelector",
    "StateBank",
    "StateBankBuilder",
    "StateBankEvaluator",
    "ExperimentAnalyzer",
    "DeterministicValidityGate",
]
