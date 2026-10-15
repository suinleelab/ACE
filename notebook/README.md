# Notebooks

`ace_mouse_tiny_example.ipynb` is the reviewer-facing notebook. It runs ACE on
`../data/ace_mouse_tiny.h5ad`, then displays the six salient representation
metrics used in the mouse droplet comparison figure.

For reproducible execution, prefer the command-line script from the repository
root:

```bash
python examples/mouse_aging_workflow.py \
  --adata data/ace_mouse_tiny.h5ad \
  --output-dir results/mouse_tiny \
  --max-epochs 100 \
  --save-all-latent \
  --skip-age-prediction
```

The notebook and script use the same core ACE calls:

1. register `AnnData` with `ACE.setup_anndata`.
2. train `ACE`.
3. extract held-out and all-cell salient/background latent representations.
4. visualize or evaluate the embeddings downstream.
