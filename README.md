# Uncertainty Estimation Thesis Appendix

This folder contains the research notebooks, helper code, calibration artifacts,
and exported evaluation data used for the uncertainty-estimation part of the
thesis. It is not the deployed `ue_service` itself. The deployed service lives
in the IntelliProcure repository and consumes the calibrated JSON artifact
exported from this analysis.

The core result documented here is a calibrated uncertainty-estimation pipeline
for GPT20B legal-check outputs:

- selected scorers: `entailment`, `noncontradiction`, `semantic_density`
- check-to-test aggregation: `gated_soft_or` in thesis (`gated_noisy_or`)
- threshold: `0.335`
- gate floor: `0.235`
- second-stage judge: `openai/gpt-oss-120b`
- selected judge review files: `gpt120b_flag_review_*_v3.csv`

The final production artifact is:

```text
evaluation/ensemble_config_threshold_0.335_gate_0.235_gpt120b_judge.json
```

## Repository Layout

```text
uncertainty_estimation/
+-- .env.example
+-- pyproject.toml
+-- uv.lock
+-- evaluation/
    +-- calibration.ipynb
    +-- calibration_helpers.py
    +-- UE.ipynb
    +-- UE_GPT20B.ipynb
    +-- ue_gpt20b_support.py
    +-- ensemble_config_threshold_0.335_gate_0.235_gpt120b_judge.json
    +-- Epic6_GT.csv
    +-- grid_search_report.md
    +-- thesis_methodology.md
    +-- thesis_implementation.md
    +-- data/
```

## Main Notebooks

### `evaluation/UE.ipynb`

Early exploratory notebook for uncertainty estimation with UQLM. It documents
and runs the first broad comparison of uncertainty methods across model families.
It includes:

- white-box scorers based on token probabilities and sampled logprobs
- black-box scorers based on sampled-answer consistency
- judge-style scorers
- cross-model data generation for GPT20B, Gemini, Qwen, and Mistral variants

This notebook is mainly the data-generation and exploration appendix for the
cross-model part of the thesis.

### `evaluation/UE_GPT20B.ipynb`

GPT20B-specific data-generation notebook. It produces repeated ensemble runs
used for the final calibration. It includes:

- single-test GPT20B ensemble runs
- all-15-tests GPT20B runs over the 13 labelled project IDs
- sampled answers and logprob capture
- NLI-based scorer computation
- UQLM workarounds for multilingual NLI label order and long-input truncation
- generated CSV/checkpoint outputs used later by `calibration.ipynb`

This notebook is the appendix source for the repeated GPT20B ensemble data.

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
- evaluates the GPT120B second-stage judge
- exports the final production configuration JSON

If reproducing the thesis tables, start here after the data files exist.

## Helper Code

### `evaluation/calibration_helpers.py`

Shared helper module for the calibration notebook. It contains the reusable
logic for:

- loading ensemble files
- q05/q95 scaling
- check-to-test aggregation
- noisy_OR and gated noft-OR
- threshold metrics such as precision, recall, F1, F2, and AP-error
- summary tables used by the calibration notebook

### `evaluation/ue_gpt20b_support.py`

Support module for `UE_GPT20B.ipynb`. It centralizes the live data-generation
helpers used during GPT20B runs:

- environment loading
- LLM backend configuration
- chat completion calls with retry/backoff
- logprob conversion into UQLM-compatible format
- prompt construction from the legal-service prompt files
- project-summary fetching
- UQLM scorer imports

## Calibration Artifact

### `evaluation/ensemble_config_threshold_0.335_gate_0.235_gpt120b_judge.json`

Minimal production configuration exported from the calibration notebook. It is
the file that should be copied into the deployed `ue_service`.

It contains:

- selected scorers
- threshold and gate floor
- aggregation rule
- q05/q95 scaling bounds
- judge decision rule
