# Reviewer-2 Defense Memo

This memo distills the four structural attacks most likely to appear in review and states the repository-supported defense for each.

## 1) The “Vacuous Metric” attack

### Attack
The flagship theorem-bearing metric,

$$
\mathrm{VLI}_t^{\mathrm{cert}} = A_t[\mathrm{LCB}_t(\hat a_t^\star)-\mathrm{UCB}_t(a_{\mathrm{LLM},t})]_+,
$$

can evaluate to zero in benign settings if the confidence radii dominate the local gap.

### Defense
The certified quantity is a **conservative lower bound**, not the sole effect-size statistic. The repository therefore reports three layers simultaneously:

1. **exact local gaps** when `exact_action_value(...)` is available,
2. **estimated and decomposed gaps** via $\mathrm{CC}$ and gated $\mathrm{RPR}$,
3. **certification power diagnostics** via `scripts/diagnose_certification_power.py`.

The critical margin is

$$
\Delta_t^{\mathrm{cert}}
=
\hat Q_t(\hat a_t^\star)-\hat Q_t(a_{\mathrm{LLM},t})
-
(\beta_t(\hat a_t^\star)+\beta_t(a_{\mathrm{LLM},t})).
$$

Certification activates if and only if $A_t=1$ and $\Delta_t^{\mathrm{cert}}>0$. A negative average margin does **not** imply the absence of reward-suboptimal but constraint-optimal divergence; it implies insufficient separation under the current rollout budget and variance profile.

## 2) The “Rules-Engine” / CMDP bait-and-switch attack

### Attack
A reviewer may argue that the benchmark merely hard-codes rules and then penalizes the LLM for leaving the feasible region, thereby stripping out semantic steerability.

### Defense
The benchmark is explicitly scoped to **topological steerability**, not unconstrained semantic persona mimicry. The object under study is not “does the model sound like a persona?” but rather:

> can a constraint-conditioned behavioral policy maintain valid topological adherence to an endogenous constraint when the projection onto the feasible set is reward-suboptimal?

This restriction is methodological. In mathematically hostile environments, soft semantic judges are not sufficiently grounded to serve as theorem-bearing validators. Deterministic endogenous constraints are the smallest auditable unit that preserves scientific identifiability.

## 3) The computational intractability attack

### Attack
The oracle audits every legal root action, making the protocol appear computationally infeasible in large branching-factor environments.

### Defense
The repository makes the complexity explicit:

$$
\mathcal{O}(|\mathcal{A}_t| \cdot n \cdot H \cdot C_{\mathrm{step}}).
$$

This is a **localized assay**, not a claim of universal full-environment scalability. The correct deployment regime is:

- moderate root branching factor,
- environment-native action abstraction,
- upstream candidate proposal or templated action generation,
- or surrogate environments in which the audited root set is intentionally tractable.

The benchmark measures identifiable constrained divergence; it is not a replacement for large-scale online planning benchmarks.

## 4) Continuation-policy bias

### Attack
The local value $Q_t(a)$ depends on the rollout continuation policy $\rho_{\mathrm{cont}}$, so the baseline may be optimal only relative to that policy.

### Defense
This is correct and should be stated explicitly. The repository therefore treats the value function as

$$
Q_t^{\rho_{\mathrm{cont}}}(a),
$$

and correspondingly

$$
\mathrm{CC}_t^{\psi,\rho_{\mathrm{cont}}},
\qquad
\mathrm{RPR}_t^{\psi,\rho_{\mathrm{cont}}}.
$$

These quantities are policy-conditioned counterfactuals, not universal truths. The oracle does not claim absolute optimality; it establishes a **fixed, auditable competence reference**. The correct robustness protocol is to rerun the exact same experiment with alternative rollout policies and report sensitivity.

## Additional code-level defenses already enforced

- Audit RNG and selector RNG are independent.
- $\mathrm{RPR}$ is undefined when $A_t=0$.
- Feasible-set collapse remains `NaN` through aggregation.
- The reference config can be replaced by a production surrogate config without changing the evaluation code.
