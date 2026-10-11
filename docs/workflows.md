# Workflows

This document summarizes the analysis workflows used with ACE. The examples are
intended as reproducible templates; manuscript-scale analyses may require more
epochs, GPUs, and dataset-specific preprocessing.

## 1. Mouse aging embedding workflow

Input:

- small included example: `data/ace_mouse_tiny.h5ad`
- raw counts in `layers["raw_counts"]`
- metadata: `age_float`, `cell_type`, `cell_type_int`, `tissue`, `sex`, `batch`

Reviewer quick run:

```bash
python examples/mouse_aging_workflow.py \
  --adata data/ace_mouse_tiny.h5ad \
  --output-dir results/mouse_tiny \
  --max-epochs 100 \
  --save-all-latent
```

For manuscript-scale analyses, use the same command-line script with `--adata`
pointing to a user-provided processed `.h5ad` file and increase `--max-epochs`.
The workflow uses `cell_type_int`, `tissue`, and `sex` as background keys for
model training, matching the mouse droplet analysis code, and uses readable
`cell_type`, `tissue`, and `sex` annotations for plots and representation
metrics.

Outputs:

- `salient_latent.csv`
- `background_latent.csv`
- `age_prediction_metrics.csv`
- `salient_metrics_age_corr.csv`
- `salient_latent_all.h5ad` and `background_latent_all.h5ad` with
  `--save-all-latent`
- optional UMAP plots with `--plot`

Interpretation:

- salient latent space should capture age-related variation.
- background latent space should capture non-aging factors such as tissue and
  cell type.

## 2. PsychAD age-prediction workflow

Input:

- PsychAD `AnnData` with raw counts and donor metadata.
- age column such as `ageDeath_float`.
- donor identifier such as `Individual_ID`.

Typical steps:

1. split donors into train/test/leaveout sets.
2. train ACE on training cells.
3. extract salient latent embeddings for held-out cells.
4. train an age-prediction head or regression model on the salient embedding.
5. aggregate cell-level predictions to donor-level biological age gaps.
6. associate biological age gaps with clinical or neuropathological phenotypes.

The key ACE calls are the same as in the mouse workflow; only metadata keys and
downstream evaluation differ.

## 3. Cross-species workflow

Input:

- merged `AnnData` containing matched genes across species.
- species annotation in `adata.obs["species"]`.
- raw counts in `layers["raw_counts"]`.

Register species as a background factor so that aging signal can be separated
from species-specific expression differences:

```python
ACE.setup_anndata(
    adata,
    layer="raw_counts",
    continuous_phenotype_keys=["age_float"],
    categorical_background_keys=["species", "cell_type", "tissue", "sex"],
    batch_key="batch",
)
```

After training, use the salient latent space to compare aging trajectories across
species and the background latent space to inspect species, tissue, and cell-type
structure.

## 4. Baseline comparison workflow

The manuscript comparisons use the same train/test/leaveout splits and the same
gene set for all methods. The common downstream target is chronological age or a
donor-level biological age gap.

Typical baselines:

| Method | Input | Downstream predictor |
|---|---|---|
| ACE | salient latent embedding | age head or regression model |
| scVI | scVI latent embedding | age head or regression model |
| Gene-expression NN | normalized gene-expression matrix | feed-forward neural network |
| ElasticNet | normalized gene-expression matrix | ElasticNet regression |

Recommended reporting:

- Pearson correlation and RMSE for age prediction.
- donor-level biological age gap after aggregating cell predictions.
- partial R2 for phenotype association, adjusting for chronological age and
  available covariates.
- identical split files and gene sets for all compared methods.

The public ACE package implements ACE representation learning. Baseline models
are standard models and can be implemented with scvi-tools and scikit-learn using
the same processed `AnnData` files and splits.
