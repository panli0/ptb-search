# PTB-Search

Companion repository for the arXiv paper:

**Dictionaries, not Darwin: set-level selection beats the evolution loop in LLM-based equation discovery**

Author: Pan Li

## Status

The paper has been submitted to arXiv. Code, frozen configurations, result tables, figure scripts, and audit artifacts are being organized and will be released after arXiv moderation.

This placeholder repository is created so the paper can point to a stable project URL before the full code release.

## What will be released

The first public code release is planned to include:

- term-bank extraction from independent LLM proposals;
- train-only sparse set-level recombination used by PTB-Search;
- frozen configurations for the Llama and DeepSeek anchors reported in the paper;
- scripts for reproducing the main tables and figures;
- result CSVs used for official239, LSR-Synth, LSR-Transform, and program-domain probes;
- completeness and leakage audit scripts;
- notes on which metrics are train-side selection metrics and which are post-hoc OOD diagnostics.

## Reproducibility policy

The repository will separate:

- `configs/`: frozen experiment settings and prompts;
- `src/`: reusable PTB-Search implementation;
- `scripts/`: command-line reproduction entry points;
- `results/`: frozen result tables used in the paper;
- `audits/`: leakage, completeness, and consistency checks;
- `paper/`: paper-specific metadata and figure/table notes;
- `data/`: dataset preparation notes and pointers.

No test, OOD, true-formula, or oracle information is used for fitting, term-bank construction, or train-side selection in the reported PTB-Search results.

## Citation

A citation file is included in `CITATION.cff`. The arXiv identifier will be added after the paper is announced.
