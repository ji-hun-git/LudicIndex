# Reference Inventory and BibTeX Field Guide

## 1. Recommended BibTeX entry types and fields

### `@article`
Required fields:
- `author`
- `title`
- `journal`
- `year`

Recommended fields:
- `volume`
- `number`
- `pages`
- `doi`
- `url`

### `@inproceedings`
Required fields:
- `author`
- `title`
- `booktitle`
- `year`

Recommended fields:
- `series`
- `volume`
- `pages`
- `publisher`
- `doi`
- `url`

### `@book`
Required fields:
- `author` or `editor`
- `title`
- `publisher`
- `year`

Recommended fields:
- `edition`
- `address`
- `isbn`
- `url`

### `@misc`
Required fields:
- `title`
- `year`

Recommended fields:
- `author`
- `eprint`
- `archivePrefix`
- `primaryClass`
- `doi`
- `url`
- `note`

### `@techreport`
Required fields:
- `author`
- `title`
- `institution`
- `year`

Recommended fields:
- `number`
- `url`

## 2. References currently used in the manuscript

- `altman1999cmdp`
- `liu2023agentbench`
- `zhou2023webarena`
- `hafner2021benchmarking`
- `kuttler2020nethack`
- `ichter2023saycan`
- `yao2023react`
- `ouyang2022instructgpt`
- `gu2024llmjudge`
- `malik2025rewardbench2`
- `samuel2024personagym`
- `sorensen2024pluralistic`

## 3. Optional references to add later

- additional LLM-agent benchmarks,
- specific steerability benchmarks,
- reward-model or DPO papers,
- a final publication reference for the full production testbed if it changes from the toy environment.

## 4. Citation hygiene for this package

1. Use benchmark references for benchmark claims.
2. Use alignment references for framing, not for theorem support.
3. Use the codebase itself and exported CSVs for implementation claims.
4. Do not cite a tree-search paper if the code you ship uses flat rollout audit.
