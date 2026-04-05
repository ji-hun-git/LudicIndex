# Implementation, Equation Mapping, and Manuscript Utilization Report

This report is the bridge from the codebase to the paper.

## 1. Architectural flow

The implemented data flow is:

`Environment -> RolloutOracle -> OracleDecision -> Selector -> DeterministicValidityGate -> MetricEngine -> TurnLog / CSV exports`

### Environment
- File: `ludic_ai/env.py`
- Object: `POMDPEnv`
- Mathematical role: provides the grounded local decision state, legal actions, rewards, rollout bounds, and optional exact values.

### Oracle
- File: `ludic_ai/oracle.py`
- Object: `RolloutOracle`
- Mathematical role: computes $\hat Q_t(a)$, $\hat\sigma_t^2(a)$, $\mathrm{LCB}_t(a)$, and $\mathrm{UCB}_t(a)$ for every legal root action.

### Selector
- File: `ludic_ai/selectors.py`
- Objects: `BaseSelector`, `AbstractLLMMenuSelector`, `NoisyGroundedSelector`, `FreeFormRandomSelector`
- Mathematical role: emits $a_{LLM,t}$ given the audited decision state and persona.

### Persona
- File: `ludic_ai/persona.py`
- Objects: `Persona`, `AttributeQuantilePersona`, `FeatureThresholdPersona`, `CallablePersona`, `CompositePersona`
- Mathematical role: defines $F_{\psi,t}$ through deterministic inequalities over oracle-computed features.

### Validity gate
- File: `ludic_ai/validator.py`
- Object: `DeterministicValidityGate`
- Mathematical role: computes $A_t$ from parse, legality, menu-membership (optional), and persona consistency.

### Metric engine
- File: `ludic_ai/metrics.py`
- Object: `MetricEngine`
- Mathematical role: computes raw gaps, exact gaps, $\LI_t$, $\VLI_t^{\mathrm{op}}$, $\VLI_t^{\mathrm{cert}}$, $\CC_t^\psi$, and gated $\RPR_t^\psi$.

### Evaluation and state-bank machinery
- Files: `ludic_ai/evaluation.py`, `ludic_ai/state_bank.py`
- Mathematical role: online episodes and matched-state evaluation.

### Record export and statistics
- Files: `ludic_ai/records.py`, `ludic_ai/stats.py`
- Mathematical role: preserve undefined quantities through aggregation and produce the trend tables used in the paper.

## 2. Code-to-equation map

### Local action value estimate
- Equation: $\hat Q_t(a)$
- Code: `ActionAudit.mean`
- Produced in: `RolloutOracle._audit_action()`

### Empirical variance
- Equation: $\hat\sigma_t^2(a)$
- Code: `ActionAudit.var`
- Produced in: `RolloutOracle._audit_action()`

### Empirical Bernstein bounds
- Equations: $\mathrm{LCB}_t(a)$, $\mathrm{UCB}_t(a)$
- Code: `ActionAudit.lcb`, `ActionAudit.ucb`
- Produced in: `RolloutOracle._confidence_interval()`

### Oracle reference action
- Estimated action: $\hat a_t^\star$
- Code: `OracleDecision.oracle_action_est`
- Produced in: `RolloutOracle._decision_from_audits()`

### Exact oracle action
- Exact action: $a_t^\star$
- Code: `OracleDecision.oracle_action_true`
- Produced when all actions expose `exact_value`

### Persona-constrained oracle action
- Estimated action: $\hat a_t^{\psi,\star}$
- Code: `OracleDecision.persona_action_est`
- Produced in: `RolloutOracle._decision_from_audits()`

### Feasible set
- Equation: $F_{\psi,t}$
- Code: `persona.feasible_actions(audits, history)`

### Validity gate
- Equation: $A_t$
- Code: `ValidityResult.overall`
- Produced in: `DeterministicValidityGate.validate()`

### True gap
- Equation: $Q_t(a_t^\star)-Q_t(a_{LLM,t})$
- Code: `MetricBundle.gap_true`
- Produced in: `MetricEngine.compute()`

### Raw Ludic Index
- Equation: $\LI_t = [\widehat{\mathrm{Gap}}_t]_+$
- Code: `MetricBundle.li`
- Produced in: `MetricEngine.compute()`

### Certified Validated Ludic Index
- Equation: $\VLI_t^{\mathrm{cert}} = A_t [\mathrm{LCB}_t(\hat a_t^\star) - \mathrm{UCB}_t(a_{LLM,t})]_+$
- Code: `MetricBundle.vli_cert`
- Produced in: `MetricEngine.compute()`

### Constraint cost
- Equation: $\CC_t^\psi = Q_t(a_t^\star)-Q_t(a_t^{\psi,\star})$
- Code: `MetricBundle.cc_true`
- Produced in: `MetricEngine.compute()`

### Residual persona regret
- Equation: $\RPR_t^\psi = Q_t(a_t^{\psi,\star})-Q_t(a_{LLM,t})$
- Code: `MetricBundle.rpr_true`
- Produced in: `MetricEngine.compute()` only when `validity.overall` is true

## 3. How to use the code in the manuscript

### Introduction
Use the code architecture to motivate the phrase:

- oracle for competence,
- selector for constrained expression,
- deterministic gate for interpretability.

### Method section
Explicitly map these files to the math:

- `oracle.py` for local value estimation and confidence bounds,
- `persona.py` for deterministic feasible sets,
- `validator.py` for $A_t$,
- `metrics.py` for the decomposition.

### Experimental Setup
Use `state_bank.py` and `evaluation.py` to justify the matched-state protocol and the audit-selection RNG separation.

### Results
Use these exported tables:

- `confound_rates.csv` for the epistemic-disentanglement result,
- `pressure_trends.csv` for the pressure-to-constraint-cost result,
- `capability_trends.csv` for the capability-to-residual-regret result,
- `pressure_response_glm.csv` for the pressure-response appendix figure.

### Discussion
Use `records.py` to justify the statement that feasible-set collapse is excluded from $\CC$ aggregation rather than zero-filled.

## 4. Important manuscript-safe wording

Say:

- rollout-audited oracle
- competence-controlled evaluation
- validated persona-conditioned deviation
- gated residual persona regret
- empirical-Bernstein-certified local sacrifice

Do not say:

- intention detector
- solved game-playing system
- deterministic MCTS oracle
- proof of personality
