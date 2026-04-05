# Super Simple Execution and Analysis Guide

This guide is intentionally operational. Copy-paste the commands in order.

## 1. Install everything

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Conda alternative:

```bash
conda env create -f environment.yml
conda activate ludic-ai
```

## 2. Verify the package

```bash
PYTHONPATH=. python tests/smoke_test.py
```

Expected terminal output:

```text
smoke test passed
```

## 3. Run the reference demo

```bash
PYTHONPATH=. python examples/demo_run.py
```

This writes fresh CSV outputs under:

- `examples/demo_outputs/turns.csv`
- `examples/demo_outputs/episodes.csv`
- `examples/demo_outputs/confound_rates.csv`
- `examples/demo_outputs/pressure_trends.csv`
- `examples/demo_outputs/capability_trends.csv`
- `examples/demo_outputs/paired_tests.csv`
- `examples/demo_outputs/pressure_response_glm.csv`
- `examples/demo_outputs/state_bank/turns.csv`
- `examples/demo_outputs/state_bank/episodes.csv`

A copy of patched reference outputs is also shipped under:

- `results/reference_outputs/`

Use the shipped outputs if you want to regenerate paper figures without rerunning the benchmark.

## 4. Generate the manuscript figures

```bash
PYTHONPATH=. python scripts/plot_publication_figures.py \
  --results_dir results/reference_outputs \
  --out_dir paper/figures
```

Generated figure files:

- `paper/figures/confound_collapse.pdf`
- `paper/figures/cc_vs_pressure.pdf`
- `paper/figures/rpr_vs_capability.pdf`
- `paper/figures/pressure_response.pdf`
- `paper/generated/reference_summary_table.tex`

## 5. Compile the paper

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
cd ..
```

The PDF will appear at:

- `paper/main.pdf`

## 6. What the main CSV files mean

### `turns.csv`
Per-decision log. This is the finest-grained table.

Key columns:

- `A_t` - final validity gate
- `gap_true`, `gap_est` - exact and estimated local value gaps
- `li` - raw positive-part estimated gap
- `vli_op`, `vli_cert` - validated and certified local sacrifice
- `cc_true`, `cc_est` - constraint cost
- `rpr_true`, `rpr_est` - residual persona regret, now gated by `A_t`
- `confounded_positive_gap` - whether a positive raw gap was actually invalid

### `episodes.csv`
Seed-level aggregates.

Key columns:

- `episode_return`
- `confound_rate`
- `sum_li`, `sum_vli_op`, `sum_vli_cert`
- `sum_cc_true`, `sum_cc_est`
- `sum_rpr_true`, `sum_rpr_est`
- `mean_pressure_true`, `mean_pressure_est`

### `confound_rates.csv`
Use this for the epistemic-disentanglement claim.

You want:

- high `confound_rate` for ungrounded selectors,
- near-zero `confound_rate` for grounded selectors.

### `pressure_trends.csv`
Use this for the claim that persona pressure increases constraint cost.

You want:

- `ols_slope > 0`
- `spearman_rho > 0`

for each persona family with multiple pressure levels.

### `capability_trends.csv`
Use this for the claim that capability reduces residual persona regret.

You want:

- `ols_slope < 0`
- `spearman_rho < 0`

for each fixed persona-pressure condition.

## 7. How to interpret the outputs

### Claim 1: Grounding solves the identifiability confound

Inspect `confound_rates.csv`.

- If `confound_rate` stays high, the benchmark is still mixing invalid behavior with constrained behavior.
- If `confound_rate` collapses to zero for grounded selectors, then positive gaps become interpretable.

### Claim 2: Constraint cost is environment-driven

Inspect `state_bank/episodes.csv` or `pressure_trends.csv`.

- As `persona_pressure` rises, `sum_cc_true` or `sum_cc_est` should rise.
- This means the persona itself is becoming more expensive, independently of selector skill.

### Claim 3: Residual persona regret is capability-driven

Inspect `state_bank/episodes.csv` or `capability_trends.csv`.

- As `selector_capability` rises, `sum_rpr_true` should fall.
- This means stronger selectors are better at finding the best action inside the persona-feasible set.

### Claim 4: Certified sacrifice is conservative

Inspect `sum_vli_cert` or `vli_cert`.

- Positive values are the strongest evidence because they are both validated and confidence-bounded.
- If `vli_cert` is all zeros, that means the current rollout budget / bound width is too conservative for certification, not that the decomposition is invalid.

## 8. One-line summary command

```bash
PYTHONPATH=. python scripts/summarize_results.py --results_dir results/reference_outputs
```
