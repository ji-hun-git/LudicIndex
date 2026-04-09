# Ludic Index

**An endogenously grounded evaluation protocol for identifiable constraint-conditioned suboptimization in stochastic environments.**

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="License" src="https://img.shields.io/badge/license-research--artifact-lightgrey">
  <img alt="Status" src="https://img.shields.io/badge/status-reviewer2--hardened-success">
</p>

## Abstract

Standard evaluations collapse intrinsic behavioral adherence and extrinsic reward maximization into a single, confounded scalar. In stochastic, partially observed environments, this makes reward-suboptimal behavior uninterpretable: when a policy exhibits reward-suboptimal but constraint-optimal divergence, outcome-only logging cannot distinguish arithmetic failure, illegal action invention, and valid topological adherence to an endogenous constraint. **Ludic AI** addresses this with a competence-controlled evaluation stack. A rollout oracle audits legal root actions, computes local value estimates and empirical Bernstein confidence intervals, and emits an audited decision object. A constraint-conditioned behavioral policy then selects only among grounded alternatives. A deterministic validity gate accepts the move only if it is parsable, legal, and topologically constraint-consistent. The framework logs the local value gap, the **constraint cost** (the irreducible, environment-driven utility penalty of projecting the policy onto the feasible set), the **residual persona regret** (the avoidable optimization gap strictly within the constraint boundary), and the certified **Validated Ludic Index**.

---

## Why this repository exists

This repository is the artifact accompanying the paper on **Certified Validated Ludic Index**. Its purpose is not to build a general-purpose game-playing agent. Its purpose is to provide a **measurement protocol** in which the cause of a local reward loss is identifiable.

The benchmark enforces four separations:

1. **Competence vs. expression**: the oracle estimates value; the policy only chooses behavior.
2. **Validity vs. invalidity**: only parsable, legal, constraint-consistent moves count toward the validated metrics.
3. **Constraint cost vs. within-constraint error**: the framework distinguishes environment-driven sacrifice from selector-driven regret.
4. **Audit randomness vs. selection randomness**: rollout computation and policy sampling use independent RNG streams.

---

## Mathematical objects

Let the environment be a local stochastic decision process with grounded state $x_t$, legal root actions $\mathcal{A}_t$, and continuation policy $\rho_{\mathrm{cont}}$ used by the oracle rollouts.

### Local value

$$
Q_t^{\rho}(a)
=
\mathbb{E}\left[\sum_{k=0}^{H-1}\gamma^k r_{t+k} \mid x_t, a_t=a, \rho_{\mathrm{cont}}\right].
$$

The superscript emphasizes that all local value claims are **relative to the chosen rollout continuation policy**. This repository does not claim continuation-policy-free optimality.

### Oracle references

The empirical oracle-best action is

$$
\hat a_t^\star \in \arg\max_{a \in \mathcal{A}_t} \hat Q_t(a).
$$

If the environment provides exact local values, the exact oracle-best action is

$$
a_t^\star \in \arg\max_{a \in \mathcal{A}_t} Q_t(a).
$$

### Endogenous constraints

A constraint $\psi$ is a deterministic predicate over oracle-computed action features $z_t(a)$, inducing a feasible set

$$
F_{\psi,t}
=
\left\{a \in \mathcal{A}_t : c_{\psi,j}(z_t(a), h_t) \le 0 \ \forall j\right\}.
$$

This is deliberate. The benchmark studies **topological steerability** under externally auditable constraints. It is not a semantic free-form persona benchmark.

### Deterministic gate

The validity gate is

$$
A_t
=
\mathbb{I}[\text{parse}_t=1]\cdot
\mathbb{I}[\text{legal}_t=1]\cdot
\mathbb{I}[\text{constraint}_t=1].
$$

Only when $A_t=1$ does the framework interpret the move as valid topological adherence to an endogenous constraint.

### Constraint Cost and Residual Persona Regret

If $F_{\psi,t} \neq \varnothing$, define

$$
a_t^{\psi,\star} \in \arg\max_{a \in F_{\psi,t}} Q_t(a).
$$

Then

$$
Q_t(a_t^\star) - Q_t(a_{\mathrm{LLM},t})
=
\underbrace{Q_t(a_t^\star)-Q_t(a_t^{\psi,\star})}_{\mathrm{CC}_t^\psi}
+
\underbrace{Q_t(a_t^{\psi,\star})-Q_t(a_{\mathrm{LLM},t})}_{\mathrm{RPR}_t^\psi}.
$$

- **Constraint cost** quantifies the irreducible, environment-driven utility penalty of projecting the policy onto the feasible set.
- **Residual persona regret** measures the avoidable optimization gap strictly within the constraint boundary.

Two implementation rules are enforced:

1. If $F_{\psi,t}=\varnothing$, then $\mathrm{CC}_t^\psi$ and $\mathrm{RPR}_t^\psi$ are logged as `NaN`, never as `0.0`.
2. If $A_t=0$, then $\mathrm{RPR}_t^\psi$ is undefined and logged as `NaN`.

### Empirical Bernstein certification

For bounded rollout returns in $[L_t,U_t]$ with width $B_t=U_t-L_t$, the oracle uses

$$
\beta_t(a,\delta)
=
\hat\sigma_t(a)\sqrt{\frac{2\log(3|\mathcal{A}_t|/\delta)}{n_t(a)}}
+
\frac{3B_t\log(3|\mathcal{A}_t|/\delta)}{n_t(a)}.
$$

The confidence interval is

$$
\mathrm{LCB}_t(a)=\hat Q_t(a)-\beta_t(a,\delta),
\qquad
\mathrm{UCB}_t(a)=\hat Q_t(a)+\beta_t(a,\delta).
$$

The certified Validated Ludic Index is

$$
\mathrm{VLI}_t^{\mathrm{cert}}
=
A_t\left[\mathrm{LCB}_t(\hat a_t^\star)-\mathrm{UCB}_t(a_{\mathrm{LLM},t})\right]_+.
$$

A useful equivalent margin form is

$$
\Delta_t^{\mathrm{cert}}
=
\hat Q_t(\hat a_t^\star)-\hat Q_t(a_{\mathrm{LLM},t})
-
\bigl(\beta_t(\hat a_t^\star)+\beta_t(a_{\mathrm{LLM},t})\bigr),
$$

so that certification activates only when $\Delta_t^{\mathrm{cert}} > 0$ and $A_t=1$.

---

## Repository layout

```text
.
├── configs/
│   ├── reference_demo.yaml
│   └── production_surrogate_template.yaml
├── docs/
│   ├── REVIEWER2_DEFENSES.md
│   ├── SURROGATE_SWAP_GUIDE.md
│   ├── IMPLEMENTATION_EQUATION_MANUSCRIPT_REPORT.md
│   └── ...
├── examples/
│   ├── toy_treasure_pomdp.py
│   └── demo_run.py
├── ludic_ai/
│   ├── env.py
│   ├── oracle.py
│   ├── persona.py
│   ├── selectors.py
│   ├── validator.py
│   ├── metrics.py
│   ├── records.py
│   ├── evaluation.py
│   ├── state_bank.py
│   ├── stats.py
│   ├── types.py
│   └── __init__.py
├── paper/
│   ├── main.tex
│   ├── reviewer2_hardened_sections.tex
│   └── references.bib
├── results/
├── scripts/
│   ├── run_from_config.py
│   ├── diagnose_certification_power.py
│   ├── plot_publication_figures.py
│   └── summarize_results.py
├── surrogates/
│   └── README.md
└── tests/
    └── smoke_test.py
```

---

## Installation

### Option A: pip

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

### Option B: conda

```bash
conda env create -f environment.yml
conda activate ludic-ai
python -m pip install -e .
```

### Sanity check

```bash
PYTHONPATH=. python tests/smoke_test.py
```

---

## Run the reference artifact

### 1) End-to-end evaluation from YAML

```bash
PYTHONPATH=. python scripts/run_from_config.py --config configs/reference_demo.yaml
```

This writes:

```text
results/reference_outputs/
├── turns.csv
├── episodes.csv
├── confound_rates.csv
├── pressure_trends.csv
├── capability_trends.csv
├── paired_tests.csv
├── pressure_response_glm.csv
└── state_bank/
    ├── turns.csv
    ├── episodes.csv
    ├── confound_rates.csv
    ├── pressure_trends.csv
    ├── capability_trends.csv
    ├── paired_tests.csv
    └── pressure_response_glm.csv
```

### 2) Diagnose certification power

```bash
PYTHONPATH=. python scripts/diagnose_certification_power.py \
  --turns results/reference_outputs/state_bank/turns.csv
```

This produces `certification_power.csv` and reports whether the certification margin is actually positive. This diagnostic exists because $\mathrm{VLI}^{\mathrm{cert}}$ is intentionally conservative and can become inactive when the confidence radii dominate the empirical gap.

### 3) Generate manuscript figures

```bash
PYTHONPATH=. python scripts/plot_publication_figures.py \
  --results_dir results/reference_outputs \
  --out_dir paper/figures
```

---

## Swap the toy environment for your production surrogate

The next move is to replace the toy environment with your audited production surrogate and rerun the identical pipeline.

### Step 1: implement the surrogate

Add a module such as:

```text
surrogates/your_production_env.py
```

That module must expose a callable

```python
build_env(**kwargs) -> POMDPEnv
```

and return an object implementing the `POMDPEnv` interface.

### Step 2: point the config at the new environment

Edit `configs/production_surrogate_template.yaml`:

```yaml
environment:
  factory: surrogates.your_production_env:build_env
```

### Step 3: rerun the exact same pipeline

```bash
PYTHONPATH=. python scripts/run_from_config.py \
  --config configs/production_surrogate_template.yaml

PYTHONPATH=. python scripts/diagnose_certification_power.py \
  --turns results/production_surrogate_run/state_bank/turns.csv

PYTHONPATH=. python scripts/plot_publication_figures.py \
  --results_dir results/production_surrogate_run \
  --out_dir paper/figures/production_surrogate
```

Nothing in the evaluation code needs to change. Only the environment factory path and surrogate-specific kwargs change.

---

## How to interpret the outputs

### `confound_rates.csv`

Use this to test the epistemic disentanglement claim.

- High `confound_rate` for ungrounded baselines means positive gaps are mostly invalid.
- Near-zero `confound_rate` for grounded selectors means the benchmark has separated invalidity from valid divergence.

### `pressure_trends.csv`

Use this to test whether constraint pressure increases environment-driven cost.

- Positive slope or positive Spearman correlation supports
  $$
  \mathbb{E}[\mathrm{CC}_t^\psi] \uparrow \text{ as constraint pressure increases.}
  $$

### `capability_trends.csv`

Use this to test whether selector capability reduces within-constraint regret.

- Negative slope or negative Spearman correlation supports
  $$
  \mathbb{E}[\mathrm{RPR}_t^\psi \mid A_t=1] \downarrow \text{ as selector capability increases.}
  $$

### `pressure_response_glm.csv`

Use this to model

$$
\Pr(A_t=1 \mid \mathrm{CC}_t^\psi).
$$

A negative pressure coefficient means high constraint pressure makes valid adherence harder. A positive capability coefficient means stronger selectors remain valid longer under pressure.

### `certification_power.csv`

Use this to preempt the “vacuous metric” attack.

- `certification_rate_given_positive_true_gap` measures how often the certified metric activates when an exact positive local gap exists.
- `mean_cert_margin_given_positive_true_gap` reports how far the empirical gap is from exceeding the combined confidence radii.
- If the rate is low and the mean margin is negative, the correct conclusion is that the lower bound is too conservative under the chosen rollout budget—not that constraint-induced divergence is absent.

---

## Scope conditions and reviewer-facing limitations

### 1) This is not a soft semantic persona benchmark

The repository evaluates **constraint-conditioned behavioral policies** under **deterministic endogenous constraints**. That is a methodological choice, not a bug. Soft semantic “persona scoring” is not theorem-bearing in mathematically hostile environments; deterministic topological adherence is.

### 2) Oracle scaling is local and explicit

The exhaustive root-action audit has complexity

$$
\mathcal{O}(|\mathcal{A}_t| \cdot n \cdot H \cdot C_{\mathrm{step}}).
$$

This artifact is intended for environments with tractable local branching, environment-native action abstraction, or upstream candidate proposal. It is an **assay**, not a universal full-environment runtime benchmark.

### 3) All value claims are continuation-policy-relative

The benchmark estimates

$$
Q_t^{\rho_{\mathrm{cont}}}(a), \quad \mathrm{CC}_t^{\psi,\rho_{\mathrm{cont}}}, \quad \mathrm{RPR}_t^{\psi,\rho_{\mathrm{cont}}}.
$$

These are relative to the chosen continuation policy. The correct robustness procedure is to rerun the pipeline with multiple rollout policies and report sensitivity, not to claim continuation-policy invariance.

---

## Suggested citation framing

If you use this repository in a manuscript, describe it as:

> an endogenously grounded evaluation protocol for identifiable constraint-conditioned suboptimization, in which an audited local oracle establishes competence, a constraint-conditioned behavioral policy selects among grounded alternatives, and deterministic validity filtering yields exact, estimated, and certified measures of reward-suboptimal but constraint-optimal divergence.

---

## Where to look next

- `paper/reviewer2_hardened_sections.tex` for the submission-ready section rewrite.
- `docs/REVIEWER2_DEFENSES.md` for the four reviewer attacks and their methodological defenses.
- `docs/SURROGATE_SWAP_GUIDE.md` for integrating your production surrogate and rerunning the full pipeline.
