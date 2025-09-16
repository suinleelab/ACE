"""Model class for ACE for single cell expression data."""

import logging
import warnings
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=UserWarning,
    module=r"pytorch_lightning\.utilities\.imports",
)

from functools import partial
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from .._constants import REGISTRY_KEYS
from scvi._compat import Literal
from scvi.data import AnnDataManager
from scvi.distributions import ZeroInflatedNegativeBinomial
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
)
from scvi.dataloaders import AnnDataLoader
from scvi.model._utils import (
    _get_batch_code_from_category,
    _init_library_size,
    scrna_raw_counts_properties,
)
from scvi.model.base import UnsupervisedTrainingMixin, VAEMixin, ArchesMixin, RNASeqMixin, BaseModelClass
from scvi.model.base._utils import _de_core
from scvi.utils import setup_anndata_dsp

from ace.module.ace import ACEModule

logger = logging.getLogger(__name__)
Number = Union[int, float]


class ACEModel(
    RNASeqMixin,
    VAEMixin,
    ArchesMixin,
    UnsupervisedTrainingMixin,
    BaseModelClass):
    """
    Model class for ACE.
    Args:
    ----
        adata: AnnData object that has been registered via
            `ACEModel.setup_anndata`.
        n_batch: Number of batches. If 0, no batch effect correction is performed.
        n_hidden: Number of nodes per hidden layer.
        n_latent: Dimensionality of the latent space.
        n_layers: Number of hidden layers used for encoder and decoder NNs.
        dropout_rate: Dropout rate for neural networks.
        use_observed_lib_size: Use observed library size for RNA as scaling factor in
            mean of conditional distribution.
        disentangle: Whether to disentangle the salient and background latent variables.
        use_mmd: Whether to use the maximum mean discrepancy loss to force background
            latent variables to have the same distribution for background and target
            data.
        mmd_weight: Weight used for the MMD loss.
        gammas: Gamma parameters for the MMD loss.
    """

    def __init__(
        self,
        adata: AnnData,
        n_batch: int = 0,
        n_hidden: int = 128,
        n_background_latent: int = 10,
        n_salient_latent: int = 10,
        n_layers: int = 1,
        dropout_rate_encoder: float = 0.1,
        dropout_rate_pheno: float = 0.1,
        dropout_rate_back: float = 0.1, 
        use_observed_lib_size: bool = True,
        pheno_continuous_recon_penalty: float = 1,
        pheno_categorical_recon_penalty: float = 1,
        back_continuous_recon_penalty: float = 1,
        back_categorical_recon_penalty: float = 1,
        hsic_loss_penalty: float = 1, 
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
        latent_distribution: Literal["normal", "ln"] = "normal",
        **model_kwargs,       
    ) -> None:
        super().__init__(adata)

        self.latent_data_type = None
        n_cats_per_cov = (
            self.adata_manager.get_state_registry(
                REGISTRY_KEYS.CAT_COVS_KEY
            ).n_cats_per_key
            if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
            else None
        )
        n_batch = self.summary_stats.n_batch
        use_size_factor_key = (
            REGISTRY_KEYS.SIZE_FACTOR_KEY in self.adata_manager.data_registry
        )
        library_log_means, library_log_vars = None, None
        if not use_size_factor_key:
            library_log_means, library_log_vars = _init_library_size(
                self.adata_manager, n_batch
            )

        n_categorical_pheno = (
            len(self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_PHENOS_KEY).field_keys)
            if REGISTRY_KEYS.CAT_PHENOS_KEY in self.adata_manager.data_registry
            else 0
        )
        n_categorical_per_pheno = (
            self.adata_manager.get_state_registry(
                REGISTRY_KEYS.CAT_PHENOS_KEY
            ).n_cats_per_key
            if REGISTRY_KEYS.CAT_PHENOS_KEY in self.adata_manager.data_registry
            else None
        )
        n_continuous_pheno = (
            len(self.adata_manager.get_state_registry(REGISTRY_KEYS.CONT_PHENOS_KEY).columns)
            if REGISTRY_KEYS.CONT_PHENOS_KEY in self.adata_manager.data_registry
            else 0
        )

        n_categorical_back = (
            len(self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_BACKS_KEY).field_keys)
            if REGISTRY_KEYS.CAT_BACKS_KEY in self.adata_manager.data_registry
            else 0
        )
        n_categorical_per_back = (
            self.adata_manager.get_state_registry(
                REGISTRY_KEYS.CAT_BACKS_KEY
            ).n_cats_per_key
            if REGISTRY_KEYS.CAT_BACKS_KEY in self.adata_manager.data_registry
            else None
        )
        n_continuous_back = (
            len(self.adata_manager.get_state_registry(REGISTRY_KEYS.CONT_BACKS_KEY).columns)
            if REGISTRY_KEYS.CONT_BACKS_KEY in self.adata_manager.data_registry
            else 0
        )

        print("n_categorical_pheno", n_categorical_pheno)
        print("n_categorical_per_pheno", n_categorical_per_pheno)
        print("n_continuous_pheno", n_continuous_pheno)
        print("n_categorical_back", n_categorical_back)
        print("n_categorical_per_back", n_categorical_per_back)
        print("n_continuous_back", n_continuous_back)

        self.module = ACEModule(
            n_input=self.summary_stats["n_vars"],
            n_batch=n_batch,
            n_labels=self.summary_stats.n_labels,
            n_hidden=n_hidden,
            n_background_latent=n_background_latent,
            n_salient_latent=n_salient_latent,
            n_continuous_cov=self.summary_stats.get("n_extra_continuous_covs", 0),
            n_cats_per_cov=n_cats_per_cov,
            n_categorical_pheno=n_categorical_pheno,
            n_categorical_per_pheno=n_categorical_per_pheno,
            n_continuous_pheno=n_continuous_pheno,
            n_categorical_back=n_categorical_back,
            n_categorical_per_back=n_categorical_per_back,
            n_continuous_back=n_continuous_back,
            n_layers=n_layers,
            dropout_rate_encoder=dropout_rate_encoder,
            dropout_rate_pheno=dropout_rate_pheno,
            dropout_rate_back=dropout_rate_back,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution,
            use_observed_lib_size=use_observed_lib_size,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            pheno_continuous_recon_penalty=pheno_continuous_recon_penalty,
            pheno_categorical_recon_penalty=pheno_categorical_recon_penalty,
            back_continuous_recon_penalty=back_continuous_recon_penalty,
            back_categorical_recon_penalty=back_categorical_recon_penalty,
            hsic_loss_penalty=hsic_loss_penalty,
            latent_data_type=self.latent_data_type,
            **model_kwargs,
        )
        self._model_summary_string = "ACE."
        # Necessary line to get params to be used for saving and loading.
        self.init_params_ = self._get_init_params(locals())
        logger.info("The model has been initialized")

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: Optional[str] = None,
        categorical_phenotype_keys: Optional[List[str]] = None,
        continuous_phenotype_keys: Optional[List[str]] = None,
        categorical_background_keys: Optional[List[str]] = None,
        continuous_background_keys: Optional[List[str]] = None,
        batch_key: Optional[str] = None,
        labels_key: Optional[str] = None,
        size_factor_key: Optional[str] = None,
        categorical_covariate_keys: Optional[List[str]] = None,
        continuous_covariate_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        %(summary)s.
        Parameters
        ----------
        %(param_layer)s
        %(param_batch_key)s
        %(param_labels_key)s
        %(param_size_factor_key)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalJointObsField(
                REGISTRY_KEYS.CAT_PHENOS_KEY, categorical_phenotype_keys
            ),
            NumericalJointObsField(
                REGISTRY_KEYS.CONT_PHENOS_KEY, continuous_phenotype_keys
            ),
            CategoricalJointObsField(
                REGISTRY_KEYS.CAT_BACKS_KEY, categorical_background_keys
            ),
            NumericalJointObsField(
                REGISTRY_KEYS.CONT_BACKS_KEY, continuous_background_keys
            ),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            NumericalObsField(
                REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key , required=False
            ),
            CategoricalJointObsField(
                REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys
            ),
            NumericalJointObsField(
                REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys
            ),
        ]
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    @torch.no_grad()
    def get_latent_representation(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        give_mean: bool = True,
        batch_size: Optional[int] = None,
        representation_kind: str = "salient",
    ) -> np.ndarray:
        """
        Return the background or salient latent representation for each cell.

        Args:
        ----
        adata: AnnData object with equivalent structure to initial AnnData. If `None`,
            defaults to the AnnData object used to initialize the model.
        indices: Indices of cells in adata to use. If `None`, all cells are used.
        give_mean: Give mean of distribution or sample from it.
        batch_size: Mini-batch size for data loading into model. Defaults to
            `scvi.settings.batch_size`.
        representation_kind: Either "background" or "salient" for the corresponding
            representation kind.

        Returns
        -------
            A numpy array with shape `(n_cells, n_latent)`.
        """
        available_representation_kinds = ["background", "salient"]
        assert representation_kind in available_representation_kinds, (
            f"representation_kind = {representation_kind} is not one of"
            f" {available_representation_kinds}"
        )

        adata = self._validate_anndata(adata)
        data_loader = self._make_data_loader(
            adata=adata,
            indices=indices,
            batch_size=batch_size,
            shuffle=False,
            data_loader_class=AnnDataLoader,
        )
        latent = []

        for tensors in data_loader:
            x = tensors[REGISTRY_KEYS.X_KEY]
            batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
            cont_key = REGISTRY_KEYS.CONT_COVS_KEY
            cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

            cat_key = REGISTRY_KEYS.CAT_COVS_KEY
            cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

            outputs = self.module.inference(
                x=x, batch_index=batch_index, cont_covs=cont_covs, cat_covs=cat_covs, n_samples=1
            )

            if representation_kind == "background":
                latent_m = outputs["qz_m"]
                latent_sample = outputs["z"]
            else:
                latent_m = outputs["qu_m"]
                latent_sample = outputs["u"]

            if give_mean:
                latent_sample = latent_m

            latent += [latent_sample.detach().cpu()]
        return torch.cat(latent).numpy()

    @torch.no_grad()
    def get_normalized_expression(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        transform_batch: Optional[Sequence[Union[Number, str]]] = None,
        gene_list: Optional[Sequence[str]] = None,
        library_size: Union[float, Literal["latent"]] = 1,
        n_samples: int = 1,
        n_samples_overall: Optional[int] = None,
        batch_size: Optional[int] = None,
        return_mean: bool = True,
        return_numpy: Optional[bool] = None,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Return the normalized (decoded) gene expression.

        Args:
        ----
        adata: AnnData object with equivalent structure to initial AnnData. If `None`,
            defaults to the AnnData object used to initialize the model.
        indices: Indices of cells in adata to use. If `None`, all cells are used.
        transform_batch: Batch to condition on. If transform_batch is:
            - None, then real observed batch is used.
            - int, then batch transform_batch is used.
        gene_list: Return frequencies of expression for a subset of genes. This can
            save memory when working with large datasets and few genes are of interest.
        library_size:  Scale the expression frequencies to a common library size. This
            allows gene expression levels to be interpreted on a common scale of
            relevant magnitude. If set to `"latent"`, use the latent library size.
        n_samples: Number of posterior samples to use for estimation.
        n_samples_overall: The number of random samples in `adata` to use.
        batch_size: Mini-batch size for data loading into model. Defaults to
            `scvi.settings.batch_size`.
        return_mean: Whether to return the mean of the samples.
        return_numpy: Return a `numpy.ndarray` instead of a `pandas.DataFrame`.
            DataFrame includes gene names as columns. If either `n_samples=1` or
            `return_mean=True`, defaults to `False`. Otherwise, it defaults to `True`.

        Returns
        -------
        If `n_samples` is provided and `return_mean` is False,
        this method returns a 3d tensor of shape (n_samples, n_cells, n_genes).
        If `n_samples` is provided and `return_mean` is True, it returns a 2d tensor
        of shape (n_cells, n_genes).
        In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        Otherwise, the method expects `n_samples_overall` to be provided and returns a 2d tensor
        of shape (n_samples_overall, n_genes).
        """
        adata = self._validate_anndata(adata)

        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            indices = np.random.choice(indices, n_samples_overall)
        data_loader = self._make_data_loader(
            adata=adata,
            indices=indices,
            batch_size=batch_size,
            shuffle=False,
            data_loader_class=AnnDataLoader,
        )

        transform_batch = _get_batch_code_from_category(
            self.get_anndata_manager(adata, required=True), transform_batch
        )

        gene_mask = slice(None) if gene_list is None else adata.var_names.isin(gene_list)

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and"
                    " return_mean is False, returning np.ndarray"
                )
            return_numpy = True
        if library_size == "latent":
            generative_output_key = "px_rate"
            scaling = 1
        else:
            generative_output_key = "px_scale"
            scaling = library_size

        exprs = []
        for tensors in data_loader:
            per_batch_exprs = []
            for batch in transform_batch:
                generative_kwargs = self._get_transform_batch_gen_kwargs(batch)
                inference_kwargs = {"n_samples": n_samples}
                _, generative_outputs = self.module.forward(
                    tensors=tensors,
                    inference_kwargs=inference_kwargs,
                    generative_kwargs=generative_kwargs,
                    compute_loss=False,
                )
                exp_ = generative_outputs[generative_output_key]
                exp_ = exp_[..., gene_mask]
                exp_ *= scaling
                per_batch_exprs.append(exp_[None].cpu())

            per_batch_exprs = torch.cat(per_batch_exprs, dim=0).mean(0).numpy()
            exprs.append(per_batch_exprs)

        cell_axis = 1 if n_samples > 1 else 0
        exprs = np.concatenate(exprs, axis=cell_axis)

        if n_samples_overall is not None:
            # Converts the 3d tensor to a 2d tensor
            exprs = exprs.reshape(-1, exprs.shape[-1])
            n_samples_ = exprs.shape[0]
            ind_ = np.random.choice(n_samples_, n_samples_overall, p=p, replace=True)
            exprs = exprs[ind_]
        elif n_samples > 1 and return_mean:
            exprs = exprs.mean(0)

        if return_numpy is None or return_numpy is False:
            return pd.DataFrame(
                exprs,
                columns=adata.var_names[gene_mask],
                index=adata.obs_names[indices],
            )
        else:
            return exprs
        
    @torch.inference_mode()
    def get_latent_library_size(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        give_mean: bool = True,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        r"""Returns the latent library size for each cell.

        This is denoted as :math:`\ell_n` in the scVI paper.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        give_mean
            Return the mean or a sample from the posterior distribution.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        """
        self._check_if_trained(warn=False)

        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
        libraries = []
        for tensors in scdl:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)

            library = outputs["library"]
            if not give_mean:
                library = torch.exp(library)
            else:
                ql = (outputs["ql_m"], outputs["ql_v"])
                if ql is None:

                    raise RuntimeError(
                        "The module for this model does not compute the posterior"
                        "distribution for the library size. Set `give_mean` to False"
                        "to use the observed library size instead."
                    )
                library = torch.distributions.LogNormal(ql[0], ql[1]).mean
            libraries += [library.cpu()]
        return torch.cat(libraries).numpy()