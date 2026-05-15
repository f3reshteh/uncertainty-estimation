# Uncertainty Estimation Thesis Appendix

This repository contains the research notebooks, helper code, and calibration
artifact used for the uncertainty-estimation part of the thesis. It is not the
deployed `ue_service` itself. The deployed service lives in the IntelliProcure
repository and consumes the calibrated JSON artifact exported from this
analysis.

The core result documented here is a calibrated uncertainty-estimation pipeline
for GPT-OSS-20B legal-check outputs:

- selected scorers: `entailment`, `noncontradiction`, `semantic_density`
- check-to-test aggregation: `gated_noisy_or`
- threshold: `0.335`
- gate floor: `0.235`
- second-stage judge: `openai/gpt-oss-120b`

The final production artifact is:

```text
evaluation/ensemble_config_threshold_0.335_gate_0.235_gpt120b_judge.json
```

## Data Availability

The `evaluation/data/` directory is excluded from this repository. It holds the
generated ensemble runs (per-test CSVs and the large all-test combined files,
including the GPT20B judge review files referenced by the notebooks). Without
it the notebooks cannot be executed end-to-end as-is; they remain readable as
documented analysis, and the calibration artifact above is the self-contained
result.

If you need the data to reproduce the analysis, contact the author.

## Repository Layout

```text
uncertainty_estimation/
├── .env.example
├── .gitignore
├── README.md
├── pyproject.toml
├── uv.lock
└── evaluation/
    ├── UE.ipynb
    ├── UE_GPT20B.ipynb
    ├── calibration.ipynb
    ├── calibration_helpers.py
    ├── ue_gpt20b_support.py
    ├── ensemble_config_threshold_0.335_gate_0.235_gpt120b_judge.json
    ├── Epic6_GT.csv
    ├── calibration_copy_thesis_report_assets/   # rendered figures from calibration.ipynb
    └── data/                                    # gitignored: generated ensemble runs
```

## Main Notebooks

### `evaluation/UE.ipynb`

Early exploratory notebook for uncertainty estimation with uqlm. Runs the first
broad comparison of uncertainty methods across model families. It includes:

- white-box scorers based on token probabilities and sampled logprobs
- black-box scorers based on sampled-answer consistency
- judge-style scorers
- cross-model data generation for GPT-OSS-20B, Gemini, Qwen, and Mistral variants

This notebook is the data-generation and exploration appendix for the
cross-model part of the thesis.

### `evaluation/UE_GPT20B.ipynb`

GPT-OSS-20B-specific data-generation notebook. Produces the repeated ensemble
runs used for the final calibration. It includes:

- single-test ensemble runs
- all-15-tests runs over the 13 labelled project IDs
- sampled-answer and logprob capture
- NLI-based scorer computation
- uqlm workarounds for multilingual NLI label order and long-input truncation
- generated CSV/checkpoint outputs used later by `calibration.ipynb`

This notebook is the appendix source for the repeated GPT-OSS-20B ensemble data.

### `evaluation/calibration.ipynb`

Main analytical calibration notebook. This is the central notebook for the
methodology and evaluation chapter. It:

- loads the generated CSV data from `evaluation/data/`
- compares white-box, black-box, and judge uncertainty families
- evaluates per-scorer AP-error
- fits scorer orientation and q05/q95 scaling bounds
- aggregates check-level scores to test-level scores
- performs scorer-subset and threshold grid search
- evaluates project-level and test-family diagnostics
- evaluates the GPT-OSS-120B second-stage judge
- exports the final production configuration JSON

If reproducing the thesis tables, start here after the data files exist.

## Helper Code

### `evaluation/calibration_helpers.py`

Shared helper module for the calibration notebook. It contains the reusable
logic for:

- loading ensemble files
- q05/q95 scaling
- check-to-test aggregation
- noisy-OR and gated noisy-OR
- threshold metrics such as precision, recall, F1, F2, and AP-error
- summary tables used by the calibration notebook

### `evaluation/ue_gpt20b_support.py`

Support module for `UE_GPT20B.ipynb`. It centralises the live data-generation
helpers used during GPT-OSS-20B runs:

- environment loading
- LLM backend configuration
- chat completion calls with retry/backoff
- logprob conversion into uqlm-compatible format
- prompt construction from the legal-service prompt files
- project-summary fetching
- uqlm scorer imports

## Calibration Artifact

### `evaluation/ensemble_config_threshold_0.335_gate_0.235_gpt120b_judge.json`

Minimal production configuration exported from the calibration notebook. It is
the file copied into the deployed `ue_service`.

It contains:

- selected scorers
- threshold and gate floor
- aggregation rule
- q05/q95 scaling bounds
- judge decision rule
