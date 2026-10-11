#!/usr/bin/env python
"""Self-contained ACE quickstart on a synthetic aging dataset.

The script generates a small count matrix with an age-associated gene program,
trains ACE, extracts salient/background embeddings, and evaluates whether the
salient embedding predicts age. It is intended as an installation smoke test and
API example, not as a biological benchmark.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ace.model import ACE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/synthetic_quickstart"))
    parser.add_argument("--n-cells", type=int, default=600)
    parser.add_argument("--n-genes", type=int, default=200)
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


def make_synthetic_adata(n_cells: int, n_genes: int, seed: int) -> anndata.AnnData:
    rng = np.random.default_rng(seed)
    age = rng.uniform(3.0, 24.0, size=n_cells).astype("float32")
    age_scaled = (age - age.mean()) / age.std()

    cell_types = np.array(["Neuron", "Astrocyte", "Microglia"])
    tissues = np.array(["cortex", "hippocampus"])
    sexes = np.array(["female", "male"])
    batches = np.array(["batch_0", "batch_1"])

    cell_type = rng.choice(cell_types, size=n_cells)
    tissue = rng.choice(tissues, size=n_cells)
    sex = rng.choice(sexes, size=n_cells)
    batch = rng.choice(batches, size=n_cells)

    base = rng.gamma(shape=1.5, scale=1.0, size=n_genes)
    age_effect = np.zeros(n_genes, dtype="float32")
    age_effect[: max(10, n_genes // 10)] = rng.normal(0.35, 0.08, size=max(10, n_genes // 10))
    age_effect[max(10, n_genes // 10) : max(20, n_genes // 5)] = rng.normal(
        -0.30, 0.08, size=max(10, n_genes // 10)
    )

    cell_type_effect = {
        key: rng.normal(0.0, 0.25, size=n_genes) for key in cell_types
    }
    tissue_effect = {
        key: rng.normal(0.0, 0.15, size=n_genes) for key in tissues
    }

    eta = np.log1p(base)[None, :] + np.outer(age_scaled, age_effect)
    eta += np.vstack([cell_type_effect[x] for x in cell_type])
    eta += np.vstack([tissue_effect[x] for x in tissue])
    eta = np.clip(eta, -4.0, 4.0)
    counts = rng.poisson(np.exp(eta)).astype("float32")

    obs = pd.DataFrame(
        {
            "age_float": age,
            "cell_type": cell_type,
            "tissue": tissue,
            "sex": sex,
            "batch": batch,
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
    adata = anndata.AnnData(X=sparse.csr_matrix(counts), obs=obs, var=var)
    adata.layers["raw_counts"] = adata.X.copy()
    return adata


def train_ace(train_adata: anndata.AnnData, args: argparse.Namespace) -> ACE:
    ACE.setup_anndata(
        train_adata,
        layer="raw_counts",
        continuous_phenotype_keys=["age_float"],
        categorical_background_keys=["cell_type", "tissue", "sex"],
        batch_key="batch",
    )

    model = ACE(
        train_adata,
        n_hidden=32,
        n_salient_latent=2,
        n_background_latent=4,
        n_layers=1,
        use_observed_lib_size=True,
        hsic_loss_penalty=1.0,
        pheno_continuous_recon_penalty=5.0,
        back_categorical_recon_penalty=[10.0, 10.0, 10.0],
        dropout_rate_encoder=0.0,
        dropout_rate_pheno=0.1,
        dropout_rate_back=0.1,
    )
    model.train(
        max_epochs=args.max_epochs,
        train_size=0.9,
        check_val_every_n_epoch=1,
        use_gpu=args.use_gpu,
        plan_kwargs={"lr": args.lr},
    )
    return model


def latent_frame(latent: np.ndarray, obs: pd.DataFrame, prefix: str) -> pd.DataFrame:
    columns = [f"{prefix}_{i}" for i in range(latent.shape[1])]
    return pd.DataFrame(latent, index=obs.index, columns=columns).join(obs)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    adata = make_synthetic_adata(args.n_cells, args.n_genes, args.seed)
    adata.write_h5ad(args.output_dir / "synthetic_adata.h5ad")

    train_idx, test_idx = train_test_split(
        np.arange(adata.n_obs), test_size=0.2, random_state=args.seed
    )
    train_adata = adata[train_idx].copy()
    test_adata = adata[test_idx].copy()

    model = train_ace(train_adata, args)
    model.save(args.output_dir / "ace_model", overwrite=True)

    train_salient = model.get_latent_representation(train_adata, representation_kind="salient")
    test_salient = model.get_latent_representation(test_adata, representation_kind="salient")
    test_background = model.get_latent_representation(test_adata, representation_kind="background")

    latent_frame(test_salient, test_adata.obs, "salient").to_csv(
        args.output_dir / "salient_latent.csv"
    )
    latent_frame(test_background, test_adata.obs, "background").to_csv(
        args.output_dir / "background_latent.csv"
    )

    reg = RidgeCV(alphas=np.logspace(-3, 3, 13))
    y_train = train_adata.obs["age_float"].to_numpy(dtype=float)
    y_test = test_adata.obs["age_float"].to_numpy(dtype=float)
    reg.fit(train_salient, y_train)
    pred = reg.predict(test_salient)
    metrics = pd.DataFrame(
        [
            {
                "n_train": train_adata.n_obs,
                "n_test": test_adata.n_obs,
                "r2": r2_score(y_test, pred),
                "rmse": mean_squared_error(y_test, pred, squared=False),
                "ridge_alpha": reg.alpha_,
            }
        ]
    )
    metrics.to_csv(args.output_dir / "age_prediction_metrics.csv", index=False)
    print(metrics.to_string(index=False))
    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
