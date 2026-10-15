#!/usr/bin/env python
"""Command-line mouse aging workflow for ACE.

This script trains ACE on a mouse aging AnnData object, extracts salient and
background latent spaces, evaluates the salient representation, and optionally
creates UMAP plots.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ace.model import ACE
from examples.evaluation_metrics import evaluate_salient_representation, save_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adata", type=Path, required=True, help="Input .h5ad file.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/mouse_aging"))
    parser.add_argument("--raw-layer", default="raw_counts")
    parser.add_argument("--age-key", default="age_float")
    parser.add_argument("--background-keys", nargs="+", default=["cell_type_int", "tissue", "sex"])
    parser.add_argument("--cell-type-key", default="cell_type")
    parser.add_argument("--cell-type-int-key", default="cell_type_int")
    parser.add_argument(
        "--evaluation-trait-keys",
        nargs="+",
        default=None,
        help=(
            "Readable annotations used for representation metrics and plots. "
            "Defaults to cell_type/tissue/sex when cell_type_int is used for training."
        ),
    )
    parser.add_argument("--batch-key", default="batch")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n-salient-latent", type=int, default=3)
    parser.add_argument("--n-background-latent", type=int, default=17)
    parser.add_argument("--hsic-loss-penalty", type=float, default=1e6)
    parser.add_argument("--pheno-continuous-recon-penalty", type=float, default=50.0)
    parser.add_argument(
        "--back-categorical-recon-penalty",
        nargs="+",
        type=float,
        default=[4000.0, 4000.0, 12000.0],
        help="One value per background key.",
    )
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--plot", action="store_true", help="Save UMAP plots with scanpy.")
    parser.add_argument("--save-all-latent", action="store_true", help="Also save latent embeddings for all cells.")
    parser.add_argument("--skip-age-prediction", action="store_true", help="Skip ridge age prediction metrics.")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip representation metrics.")
    parser.add_argument("--metric-pairs", type=int, default=5000)
    return parser.parse_args()


def prepare_annotations(adata: anndata.AnnData, args: argparse.Namespace) -> None:
    if args.cell_type_int_key not in args.background_keys:
        return
    if args.cell_type_int_key in adata.obs:
        return
    if args.cell_type_key not in adata.obs:
        raise KeyError(
            f"obs[{args.cell_type_int_key!r}] is missing and cannot be created "
            f"because obs[{args.cell_type_key!r}] is also missing."
        )
    adata.obs[args.cell_type_int_key] = LabelEncoder().fit_transform(
        adata.obs[args.cell_type_key].astype(str)
    )


def evaluation_trait_keys(adata: anndata.AnnData, args: argparse.Namespace) -> list[str]:
    if args.evaluation_trait_keys is not None:
        return args.evaluation_trait_keys
    keys = []
    for key in args.background_keys:
        if key == args.cell_type_int_key and args.cell_type_key in adata.obs:
            keys.append(args.cell_type_key)
        else:
            keys.append(key)
    return keys


def check_inputs(adata: anndata.AnnData, args: argparse.Namespace) -> None:
    missing = []
    if args.raw_layer not in adata.layers:
        missing.append(f"layers[{args.raw_layer!r}]")
    eval_keys = evaluation_trait_keys(adata, args)
    for key in [args.age_key, args.batch_key, *args.background_keys, *eval_keys]:
        if key not in adata.obs:
            missing.append(f"obs[{key!r}]")
    if missing:
        raise KeyError("Input AnnData is missing required fields: " + ", ".join(missing))
    if len(args.back_categorical_recon_penalty) != len(args.background_keys):
        raise ValueError(
            "--back-categorical-recon-penalty must contain one value per "
            "--background-key"
        )


def train_ace(train_adata: anndata.AnnData, args: argparse.Namespace) -> ACE:
    ACE.setup_anndata(
        train_adata,
        layer=args.raw_layer,
        continuous_phenotype_keys=[args.age_key],
        categorical_background_keys=args.background_keys,
        batch_key=args.batch_key,
    )

    model = ACE(
        train_adata,
        n_salient_latent=args.n_salient_latent,
        n_background_latent=args.n_background_latent,
        use_observed_lib_size=True,
        hsic_loss_penalty=args.hsic_loss_penalty,
        pheno_continuous_recon_penalty=args.pheno_continuous_recon_penalty,
        back_categorical_recon_penalty=args.back_categorical_recon_penalty,
        dropout_rate_encoder=0.0,
        dropout_rate_pheno=0.5,
        dropout_rate_back=0.5,
    )
    model.train(
        max_epochs=args.max_epochs,
        train_size=0.8,
        check_val_every_n_epoch=1,
        early_stopping=True,
        early_stopping_monitor="validation_loss",
        early_stopping_patience=45,
        use_gpu=args.use_gpu,
        plan_kwargs={"lr": args.lr},
    )
    return model


def latent_to_adata(latent: np.ndarray, obs: pd.DataFrame, prefix: str) -> anndata.AnnData:
    latent_adata = anndata.AnnData(X=latent, obs=obs.copy())
    latent_adata.var_names = [f"{prefix}_{i}" for i in range(latent.shape[1])]
    return latent_adata


def save_latent_outputs(
    model: ACE,
    adata: anndata.AnnData,
    train_adata: anndata.AnnData,
    test_adata: anndata.AnnData,
    args: argparse.Namespace,
) -> None:
    test_salient = model.get_latent_representation(test_adata, representation_kind="salient")
    test_background = model.get_latent_representation(test_adata, representation_kind="background")

    salient_adata = latent_to_adata(test_salient, test_adata.obs, "salient")
    background_adata = latent_to_adata(test_background, test_adata.obs, "background")
    salient_adata.write_h5ad(args.output_dir / "salient_latent.h5ad")
    background_adata.write_h5ad(args.output_dir / "background_latent.h5ad")

    pd.DataFrame(
        test_salient,
        index=test_adata.obs_names,
        columns=salient_adata.var_names,
    ).join(test_adata.obs).to_csv(args.output_dir / "salient_latent.csv")
    pd.DataFrame(
        test_background,
        index=test_adata.obs_names,
        columns=background_adata.var_names,
    ).join(test_adata.obs).to_csv(args.output_dir / "background_latent.csv")

    if args.save_all_latent:
        all_salient = model.get_latent_representation(adata, representation_kind="salient")
        all_background = model.get_latent_representation(adata, representation_kind="background")
        latent_to_adata(all_salient, adata.obs, "salient").write_h5ad(
            args.output_dir / "salient_latent_all.h5ad"
        )
        latent_to_adata(all_background, adata.obs, "background").write_h5ad(
            args.output_dir / "background_latent_all.h5ad"
        )

    if not args.skip_age_prediction:
        train_salient = model.get_latent_representation(train_adata, representation_kind="salient")
        reg = RidgeCV(alphas=np.logspace(-3, 3, 13))
        y_train = train_adata.obs[args.age_key].astype(float).to_numpy()
        y_test = test_adata.obs[args.age_key].astype(float).to_numpy()
        reg.fit(train_salient, y_train)
        pred = reg.predict(test_salient)
        pd.DataFrame(
            [
                {
                    "n_train": train_adata.n_obs,
                    "n_test": test_adata.n_obs,
                    "r2": r2_score(y_test, pred),
                    "rmse": mean_squared_error(y_test, pred, squared=False),
                    "ridge_alpha": reg.alpha_,
                }
            ]
        ).to_csv(args.output_dir / "age_prediction_metrics.csv", index=False)
    if not args.skip_evaluation:
        trait_keys = evaluation_trait_keys(test_adata, args)
        salient_metrics = evaluate_salient_representation(
            salient_adata,
            age_key=args.age_key,
            trait_keys=tuple(trait_keys),
            n_pairs=args.metric_pairs,
            seed=args.seed,
        )
        save_metrics(salient_metrics, args.output_dir / "salient_metrics_age_corr.csv")


def save_umap_plots(args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt
    import scanpy as sc

    for stem in ["salient_latent", "background_latent"]:
        adata = anndata.read_h5ad(args.output_dir / f"{stem}.h5ad")
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
        for key in [args.age_key, *evaluation_trait_keys(adata, args)]:
            if key not in adata.obs:
                continue
            sc.pl.umap(adata, color=key, show=False)
            plt.savefig(args.output_dir / f"{stem}_umap_{key}.png", bbox_inches="tight", dpi=300)
            plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    adata = anndata.read_h5ad(args.adata)
    prepare_annotations(adata, args)
    check_inputs(adata, args)
    adata.obs[args.age_key] = adata.obs[args.age_key].astype(float)

    train_idx, test_idx = train_test_split(
        np.arange(adata.n_obs), test_size=args.test_size, random_state=args.seed
    )
    train_adata = adata[train_idx].copy()
    test_adata = adata[test_idx].copy()

    model = train_ace(train_adata, args)
    model.save(args.output_dir / "ace_model", overwrite=True)
    save_latent_outputs(model, adata, train_adata, test_adata, args)
    if args.plot:
        save_umap_plots(args)
    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
