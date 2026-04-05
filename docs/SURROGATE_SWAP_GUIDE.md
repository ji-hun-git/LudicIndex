# Swap the Toy Environment for Your Production Surrogate

This guide explains the single intended “next move” for the repository: replace the toy reference environment with your production surrogate and rerun the identical evaluation pipeline end-to-end.

## 1) What the surrogate must implement

Your surrogate module must expose a factory:

```python
build_env(**kwargs) -> POMDPEnv
```

The returned object must implement the `POMDPEnv` interface:

- `reset(seed)`
- `clone()`
- `observe()`
- `oracle_view()` (optional override)
- `legal_actions()`
- `step(action)`
- `is_terminal()`
- `state_id()`
- `reward_range()`
- `history_context()`
- `extra_action_features(...)`
- `exact_action_value(...)` (optional but strongly recommended for exact-gap sanity checks)

## 2) Where to put the surrogate

Create a module such as:

```text
surrogates/your_production_env.py
```

with:

```python
from ludic_ai.env import POMDPEnv

class YourProductionSurrogate(POMDPEnv):
    ...


def build_env(**kwargs):
    return YourProductionSurrogate(**kwargs)
```

## 3) Point the config at the surrogate

Copy and edit:

```text
configs/production_surrogate_template.yaml
```

Set:

```yaml
environment:
  factory: surrogates.your_production_env:build_env
```

## 4) Keep the same evaluation stack

Do **not** rewrite the evaluation pipeline. The entire point of the repository is that the same stack remains valid:

```text
Environment -> Oracle -> Constraint-Conditioned Behavioral Policy -> Deterministic Gate -> Metrics -> CSVs -> Figures
```

## 5) Rerun end-to-end

```bash
PYTHONPATH=. python scripts/run_from_config.py \
  --config configs/production_surrogate_template.yaml

PYTHONPATH=. python scripts/diagnose_certification_power.py \
  --turns results/production_surrogate_run/state_bank/turns.csv

PYTHONPATH=. python scripts/plot_publication_figures.py \
  --results_dir results/production_surrogate_run \
  --out_dir paper/figures/production_surrogate
```

## 6) What to check before you trust the results

1. `confound_rate` should collapse for grounded selectors.
2. `sum_cc_true` or `sum_cc_est` should increase with constraint pressure.
3. `sum_rpr_true` or gated `sum_rpr_est` should decrease with selector capability.
4. `certification_power.csv` should show whether the empirical Bernstein certificate is actually activated.
5. Re-run with at least one alternative rollout continuation policy to quantify $\rho_{\mathrm{cont}}$ sensitivity.
