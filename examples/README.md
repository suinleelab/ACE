# ACE examples

This directory contains script-based examples. They are intended to be easier to
re-run than notebooks and to make the analysis workflow explicit.

## `synthetic_quickstart.py`

Self-contained smoke test that generates a synthetic count matrix and runs ACE.
Use this first to confirm that the package and environment are installed.

```bash
python examples/synthetic_quickstart.py \
  --output-dir results/synthetic_quickstart \
  --max-epochs 5
```

## `mouse_aging_workflow.py`

Command-line version of the mouse aging workflow. A tiny real mouse droplet
dataset is included for reviewers:

```bash
python examples/mouse_aging_workflow.py \
  --adata data/ace_mouse_tiny.h5ad \
  --output-dir results/mouse_tiny \
  --max-epochs 1
```

The reviewer quickstart uses the included tiny dataset and does not require
external data. Larger analyses require a user-provided `AnnData` file; see
[../docs/data_format.md](../docs/data_format.md) for the required columns and
layers.

Outputs include age-prediction metrics and salient representation metrics:

- `age_prediction_metrics.csv`
- `salient_metrics_age_corr.csv`
- `salient_latent.h5ad`
- `background_latent.h5ad`
