# Data format

ACE uses an `AnnData` object. Raw counts should be stored in a layer and metadata
should be stored in `adata.obs`.

## Required matrix

```python
adata.layers["raw_counts"]
```

This layer should contain non-negative raw UMI counts with shape
`n_cells x n_genes`. It may be dense or sparse. ACE registers this layer with
`is_count_data=True` and models counts using the configured likelihood.

`adata.X` may contain normalized expression for visualization or downstream
baselines, but ACE training should use the raw count layer:

```python
ACE.setup_anndata(adata, layer="raw_counts", ...)
```

## Required metadata

The exact columns depend on the workflow. Common columns are:

| Column | Type | Purpose |
|---|---|---|
| `age_float` | numeric | aging phenotype for mouse examples |
| `ageDeath_float` | numeric | aging phenotype for PsychAD examples |
| `cell_type` | categorical | background factor |
| `cell_type_int` | categorical/integer | encoded cell type used by the mouse droplet workflow |
| `tissue` | categorical | background factor |
| `sex` | categorical | background factor |
| `batch` | categorical | batch covariate |
| `donor_id` or `Individual_ID` | categorical | donor-level splitting/evaluation |

## Setup examples

Mouse aging:

```python
ACE.setup_anndata(
    adata,
    layer="raw_counts",
    continuous_phenotype_keys=["age_float"],
    categorical_background_keys=["cell_type_int", "tissue", "sex"],
    batch_key="batch",
)
```

PsychAD human brain:

```python
ACE.setup_anndata(
    adata,
    layer="raw_counts",
    continuous_phenotype_keys=["ageDeath_float"],
    categorical_background_keys=["cell_type", "sex", "Source"],
    batch_key="batch",
)
```

Cross-species analyses typically include species as a background factor:

```python
ACE.setup_anndata(
    adata,
    layer="raw_counts",
    continuous_phenotype_keys=["age_float"],
    categorical_background_keys=["species", "cell_type", "tissue", "sex"],
    batch_key="batch",
)
```

## Preprocessing notes

- Keep raw counts in `adata.layers["raw_counts"]`.
- Filter low-quality cells and genes before ACE training.
- Use consistent gene sets for comparisons across models.
- Encode continuous ages as numeric columns.
- For the mouse droplet workflow, `examples/mouse_aging_workflow.py` creates
  `cell_type_int` from `cell_type` when it is missing.
- Do not one-hot encode categorical columns manually; `setup_anndata` handles
  categorical registration.
