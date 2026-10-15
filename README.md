# ACE

ACE (Aging Cell Embedding) is a scvi-tools based deep generative model for
single-cell aging analysis. ACE learns two complementary latent spaces:

- **salient / aging latent space**: variation associated with aging phenotypes.
- **background latent space**: variation associated with technical or biological
  background factors such as cell type, tissue, sex, batch, or species.

This repository contains the ACE Python package, runnable examples, and workflow
documentation for reproducing the main analysis patterns used in the manuscript.

## Repository layout

```text
ace/                         Python package
  model/ace.py               scvi-tools style ACE model class
  module/ace.py              PyTorch module and loss implementation
examples/
  synthetic_quickstart.py    Self-contained runnable example with simulated data
  mouse_aging_workflow.py    Command-line mouse aging workflow
docs/
  data_format.md             Required AnnData fields and preprocessing notes
  workflows.md               Mouse, PsychAD, cross-species, and comparison workflows
notebook/
  ace_mouse_tiny_example.ipynb
                              Reviewer-facing runnable tiny mouse example
```

## Installation

ACE is not currently published to PyPI. Install from a local clone.

```bash
git clone https://github.com/suinleelab/ACE.git
cd ACE
conda env create -f environment.yml
conda activate ace
pip install --no-build-isolation -e .
```

The environment was tested with Python 3.9, PyTorch 1.13, scvi-tools 0.19, and
anndata 0.8. GPU training is optional but recommended for large datasets.

## Quick start with simulated data

This example creates a small synthetic count matrix, trains ACE for a few epochs,
extracts salient/background embeddings, and fits a simple age predictor on the
salient embedding.

```bash
python examples/synthetic_quickstart.py \
  --output-dir results/synthetic_quickstart \
  --max-epochs 5
```

The command creates:

```text
results/synthetic_quickstart/
  synthetic_adata.h5ad
  salient_latent.csv
  background_latent.csv
  age_prediction_metrics.csv
```

The quickstart is intentionally small and is meant to verify installation and API
usage. It is not a benchmark.

## Minimal ACE API

ACE follows the scvi-tools model pattern.

```python
import anndata
from ace.model import ACE

adata = anndata.read_h5ad("data/ace_mouse_tiny.h5ad")

adata.obs["cell_type_int"] = adata.obs["cell_type"].astype("category").cat.codes

ACE.setup_anndata(
    adata,
    layer="raw_counts",
    continuous_phenotype_keys=["age_float"],
    categorical_background_keys=["cell_type_int", "tissue", "sex"],
    batch_key="batch",
)

model = ACE(
    adata,
    n_salient_latent=3,
    n_background_latent=17,
    use_observed_lib_size=True,
    hsic_loss_penalty=1e6,
)

model.train(max_epochs=500, train_size=0.8, use_gpu=True)

salient = model.get_latent_representation(
    adata, representation_kind="salient"
)
background = model.get_latent_representation(
    adata, representation_kind="background"
)
```

Use the salient embedding for age-related downstream analyses and the background
embedding to inspect non-aging variation.

## Input data

ACE expects an `AnnData` object with raw counts stored in a count layer, typically
`adata.layers["raw_counts"]`. Phenotype and background annotations are read from
`adata.obs`.

See [docs/data_format.md](docs/data_format.md) for the complete expected format.

## Mouse aging example

The repository includes a small real mouse droplet example:

```text
data/ace_mouse_tiny.h5ad
```

This file is a 3,000-cell subset of the public mouse droplet data used in the
manuscript workflow. It contains raw counts in `layers["raw_counts"]` and the
metadata columns needed by ACE (`age_float`, `cell_type`, `cell_type_int`,
`tissue`, `sex`, and `batch`). The command-line workflow follows the manuscript
mouse droplet setup by using `cell_type_int`, `tissue`, and `sex` as background
keys for model training while keeping readable annotations for plots and
evaluation metrics.

Run the command-line workflow:

```bash
python examples/mouse_aging_workflow.py \
  --adata data/ace_mouse_tiny.h5ad \
  --output-dir results/mouse_tiny \
  --max-epochs 100 \
  --save-all-latent
```

The command creates held-out `salient_latent.h5ad` and
`background_latent.h5ad`, all-cell latent embeddings when `--save-all-latent` is
used, `age_prediction_metrics.csv`, and `salient_metrics_age_corr.csv` in the
selected output directory.

The notebook `notebook/ace_mouse_tiny_example.ipynb` runs the same ACE training
workflow with `--skip-age-prediction` and displays only the six representation
metrics used in the mouse droplet comparison figure. It is the recommended
notebook for reviewers.

The reviewer quickstart and notebook do not require any external data. Larger
analyses can use the same script with `--adata` pointing to a user-provided
`AnnData` file in the format described in [docs/data_format.md](docs/data_format.md).

## Manuscript workflows and comparisons

The manuscript analyses use the same ACE API with dataset-specific `AnnData`
objects and downstream evaluation scripts. The main workflow families are:

- mouse aging embedding analysis.
- PsychAD human brain aging and biological age-gap prediction.
- cross-species alignment of aging trajectories.
- baseline comparisons against scVI, gene-expression neural networks, ElasticNet,
  and other representation-learning methods.

See [docs/workflows.md](docs/workflows.md) for the expected inputs, high-level
commands, and how the comparison analyses are organized.

## Citation

If you use ACE, please cite the manuscript once available. Until publication,
please cite this repository and include the commit hash used for analysis.

## License

This project is distributed under the license in [LICENSE](LICENSE).
