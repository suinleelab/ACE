"""PyTorch module for Contrastive VI for single cell expression data."""
from typing import Dict, Optional, Tuple, Callable, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from .._constants import REGISTRY_KEYS
from scvi._compat import Literal
from scvi._types import LatentDataType
from scvi.distributions import ZeroInflatedNegativeBinomial, NegativeBinomial
from scvi.module.base import BaseModuleClass, LossRecorder, auto_move_data
from scvi.nn import DecoderSCVI, Decoder, Encoder, one_hot
from .scvi_my import Decoder as Decoder_my
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ace.module.utils import hsic

torch.backends.cudnn.benchmark = True


class ACEModule(BaseModuleClass):
    """
    PyTorch module for ACE.

    Args:
    ----
        n_input: Number of input genes.
        n_batch: Number of batches. If 0, no batch effect correction is performed.
        n_hidden: Number of nodes per hidden layer.
        n_background_latent: Dimensionality of the background latent space.
        n_salient_latent: Dimensionality of the salient latent space.
        n_layers: Number of hidden layers used for encoder and decoder NNs.
        dropout_rate: Dropout rate for neural networks.
        use_observed_lib_size: Use observed library size for RNA as scaling factor in
            mean of conditional distribution.
        library_log_means: 1 x n_batch array of means of the log library sizes.
            Parameterize prior on library size if not using observed library size.
        library_log_vars: 1 x n_batch array of variances of the log library sizes.
            Parameterize prior on library size if not using observed library size.
        wasserstein_penalty: Weight of the Wasserstein distance loss that further
            discourages shared variations from leaking into the salient latent space.
    """

    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_background_latent: int = 10,
        n_salient_latent: int = 10,
        n_categorical_pheno: int = 0,
        n_categorical_per_pheno: Optional[Iterable[int]] = None,
        n_continuous_pheno: int = 0,
        n_categorical_back: int = 0,
        n_categorical_per_back: Optional[Iterable[int]] = None,
        n_continuous_back: int = 0,
        n_layers: int = 1,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Optional[Iterable[int]] = None,
        dropout_rate_encoder: float = 0.1,
        dropout_rate_pheno: float = 0.1,
        dropout_rate_back: float = 0.1, 
        dispersion: str = "gene",
        log_variational: bool = True,
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
        latent_distribution: str = "normal",
        encode_covariates: bool = False,
        deeply_inject_covariates: bool = True,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_size_factor_key: bool = False,
        use_observed_lib_size: bool = True,
        library_log_means: Optional[np.ndarray] = None,
        library_log_vars: Optional[np.ndarray] = None,
        var_activation: Optional[Callable] = None,
        latent_data_type: Optional[LatentDataType] = None,
        pheno_continuous_recon_penalty: float = 1,
        pheno_categorical_recon_penalty: float = 1,
        back_continuous_recon_penalty: float = 1,
        back_categorical_recon_penalty: float = 1,
        hsic_loss_penalty: float = 1,

    ) -> None:
        super().__init__()
        self.dispersion = dispersion
        self.n_background_latent = n_background_latent
        self.n_salient_latent = n_salient_latent

        if n_categorical_pheno <=0 and n_continuous_pheno <= 0:
            raise ValueError("Must have at least one phenotype")
        self.n_categorical_pheno = n_categorical_pheno
        self.n_categorical_per_pheno = n_categorical_per_pheno
        self.n_continuous_pheno = n_continuous_pheno
        self.n_categorical_back = n_categorical_back
        self.n_categorical_per_back = n_categorical_per_back
        self.n_continuous_back = n_continuous_back

        self.log_variational = log_variational
        self.gene_likelihood = gene_likelihood
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.latent_distribution = latent_distribution
        self.encode_covariates = encode_covariates
        self.latent_data_type = latent_data_type
        self.pheno_continuous_recon_penalty = pheno_continuous_recon_penalty
        self.pheno_categorical_recon_penalty = pheno_categorical_recon_penalty
        self.back_continuous_recon_penalty = back_continuous_recon_penalty
        self.back_categorical_recon_penalty = back_categorical_recon_penalty
        self.hsic_loss_penalty = hsic_loss_penalty

        self.use_size_factor_key = use_size_factor_key
        self.use_observed_lib_size = use_size_factor_key or use_observed_lib_size
        if not self.use_observed_lib_size:
            if library_log_means is None or library_log_vars is None:
                raise ValueError(
                    "If not using observed_lib_size, "
                    "must provide library_log_means and library_log_vars."
                )

            self.register_buffer(
                "library_log_means", torch.from_numpy(library_log_means).float()
            )
            self.register_buffer(
                "library_log_vars", torch.from_numpy(library_log_vars).float()
            )

        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input))
        elif self.dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_batch))
        elif self.dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_labels))
        elif self.dispersion == "gene-cell":
            pass
        else:
            raise ValueError(
                "dispersion must be one of ['gene', 'gene-batch',"
                " 'gene-label', 'gene-cell'], but input was "
                "{}.format(self.dispersion)"
            )

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        n_input_encoder = n_input + n_continuous_cov * encode_covariates
        cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)
        encoder_cat_list = cat_list if encode_covariates else None

        # Background encoder.
        self.z_encoder = Encoder(
            n_input_encoder,
            n_background_latent,
            n_cat_list=encoder_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_encoder,
            distribution=latent_distribution,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            return_dist=False,
        )
        # Salient encoder.
        self.u_encoder = Encoder(
            n_input_encoder,
            n_salient_latent,
            n_cat_list=encoder_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_encoder,
            distribution=latent_distribution,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            return_dist=False,
        )
        # Library size encoder.
        self.l_encoder = Encoder(
            n_input_encoder,
            1,
            n_layers=1,
            n_cat_list=encoder_cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_encoder,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            return_dist=False,
        )
        # Decoder from latent variable to distribution parameters in data space.
        n_input_x_decoder = n_background_latent + n_salient_latent + n_continuous_cov
        self.x_decoder = DecoderSCVI(
            n_input_x_decoder,
            n_input,
            n_cat_list=cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
            scale_activation="softplus" if use_size_factor_key else "softmax",
        )
        n_pheno = n_continuous_pheno + (0 if n_categorical_per_pheno is None else sum(n_categorical_per_pheno))
        self.s_decoder = Decoder_my(
            n_salient_latent,
            n_pheno,
            n_cat_list=None,
            n_layers=1,
            n_hidden=16,
            dropout_rate=dropout_rate_pheno,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
        )
        n_back = n_continuous_back + (0 if n_categorical_per_back is None else sum(n_categorical_per_back))
        if n_back > 0:
            self.b_decoder = Decoder_my(
                n_background_latent,
                n_back,
                n_cat_list=None,
                n_layers=1,
                n_hidden=n_hidden,
                dropout_rate=dropout_rate_back,
                inject_covariates=deeply_inject_covariates,
                use_batch_norm=use_batch_norm_decoder,
                use_layer_norm=use_layer_norm_decoder,
            )
        else:
            self.b_decoder = None
        
        if not isinstance(pheno_continuous_recon_penalty, list):
            self.pheno_continuous_recon_penalty = torch.tensor(
                [pheno_continuous_recon_penalty for i in range(n_continuous_pheno)],
                dtype=torch.float).to(self.device)
        else:
            self.pheno_continuous_recon_penalty = torch.tensor(
                pheno_continuous_recon_penalty,
                dtype=torch.float).to(self.device)

        if not isinstance(pheno_categorical_recon_penalty, list):
            self.pheno_categorical_recon_penalty = torch.tensor(
                [pheno_categorical_recon_penalty for i in range(n_categorical_pheno)],
                dtype=torch.float).to(self.device)
        else:
            self.pheno_categorical_recon_penalty = torch.tensor(
                pheno_categorical_recon_penalty,
                dtype=torch.float).to(self.device)
        
        if not isinstance(back_continuous_recon_penalty, list):
            self.back_continuous_recon_penalty = torch.tensor(
                [back_continuous_recon_penalty for i in range(n_continuous_back)],
                dtype=torch.float).to(self.device)
        else:
            self.back_categorical_recon_penalty = torch.tensor(
                back_continuous_recon_penalty,
                dtype=torch.float).to(self.device)

        if not isinstance(back_categorical_recon_penalty, list):
            self.back_categorical_recon_penalty = torch.tensor(
                [back_categorical_recon_penalty for i in range(n_categorical_back)], 
                dtype=torch.float).to(self.device)
        else:
            self.back_categorical_recon_penalty = torch.tensor(
                back_categorical_recon_penalty,
                dtype=torch.float).to(self.device)
                
    @auto_move_data
    def _compute_local_library_params(
        self, batch_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_batch = self.library_log_means.shape[1]
        local_library_log_means = F.linear(
            one_hot(batch_index, n_batch), self.library_log_means
        )
        local_library_log_vars = F.linear(
            one_hot(batch_index, n_batch), self.library_log_vars
        )
        return local_library_log_means, local_library_log_vars

    def _get_inference_input(
        self,
        tensors,
    ):
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]

        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        if self.latent_data_type is None:
            x = tensors[REGISTRY_KEYS.X_KEY]
            input_dict = dict(
                x=x, batch_index=batch_index, cont_covs=cont_covs, cat_covs=cat_covs
            )
        else:
            if self.latent_data_type == "dist":
                qzm = tensors[REGISTRY_KEYS.LATENT_QZM_KEY]
                qzv = tensors[REGISTRY_KEYS.LATENT_QZV_KEY]
                input_dict = dict(qzm=qzm, qzv=qzv)
            else:
                raise ValueError(f"Unknown latent data type: {self.latent_data_type}")

        return input_dict

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z"]
        u = inference_outputs["u"]
        library = inference_outputs["library"]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        y = tensors[REGISTRY_KEYS.LABELS_KEY]

        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        size_factor_key = REGISTRY_KEYS.SIZE_FACTOR_KEY
        size_factor = (
            torch.log(tensors[size_factor_key])
            if size_factor_key in tensors.keys()
            else None
        )

        input_dict = dict(
            z=z,
            u=u,
            library=library,
            batch_index=batch_index,
            y=y,
            cont_covs=cont_covs,
            cat_covs=cat_covs,
            size_factor=size_factor,
        )
        return input_dict

    @staticmethod
    def _reshape_tensor_for_samples(tensor: torch.Tensor, n_samples: int):
        return tensor.unsqueeze(0).expand((n_samples, tensor.size(0), tensor.size(1)))


    @auto_move_data
    def inference(
        self,
        x: torch.Tensor,
        batch_index: torch.Tensor,
        cont_covs=None, 
        cat_covs=None, 
        n_samples: int = 1,
    ):
        """
        High level inference method.

        Runs the inference (encoder) model.
        """
        x_ = x
        if self.use_observed_lib_size:
            library = torch.log(x.sum(1)).unsqueeze(1)

        if self.log_variational:
            x_ = torch.log(1 + x_)

        if cont_covs is not None and self.encode_covariates:
            encoder_input = torch.cat((x_, cont_covs), dim=-1)
        else:
            encoder_input = x_
        if cat_covs is not None and self.encode_covariates:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()

        qz_m, qz_v, z = self.z_encoder(encoder_input, batch_index, *categorical_input)
        qu_m, qu_v, u = self.u_encoder(encoder_input, batch_index, *categorical_input)

        ql_m, ql_v = None, None
        if not self.use_observed_lib_size:
            ql_m, ql_v, library_encoded = self.l_encoder(encoder_input, batch_index, *categorical_input)
            library = library_encoded

        if n_samples > 1:
            qz_m = self._reshape_tensor_for_samples(qz_m, n_samples)
            qz_v = self._reshape_tensor_for_samples(qz_v, n_samples)
            z = self._reshape_tensor_for_samples(z, n_samples)
            qu_m = self._reshape_tensor_for_samples(qu_m, n_samples)
            qu_v = self._reshape_tensor_for_samples(qu_v, n_samples)
            u = self._reshape_tensor_for_samples(u, n_samples)

            if self.use_observed_lib_size:
                library = self._reshape_tensor_for_samples(library, n_samples)
            else:
                ql_m = self._reshape_tensor_for_samples(ql_m, n_samples)
                ql_v = self._reshape_tensor_for_samples(ql_v, n_samples)
                library = Normal(ql_m, ql_v.sqrt()).sample()

        outputs = dict(
            z=z,
            qz_m=qz_m,
            qz_v=qz_v,
            u=u,
            qu_m=qu_m,
            qu_v=qu_v,
            library=library,
            ql_m=ql_m,
            ql_v=ql_v,
        )
        return outputs

    @auto_move_data
    def _generic_generative(
        self,
        z: torch.Tensor,
        s: torch.Tensor,
        library: torch.Tensor,
        batch_index: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        latent = torch.cat([z, s], dim=-1)
        px_scale, px_r, px_rate, px_dropout = self.decoder(
            self.dispersion,
            latent,
            library,
            batch_index,
        )
        px_r = torch.exp(self.px_r)
        return dict(
            px_scale=px_scale, px_r=px_r, px_rate=px_rate, px_dropout=px_dropout
        )
        
    @auto_move_data
    def generative(
        self,
        z: torch.Tensor,
        u: torch.Tensor,
        library: torch.Tensor,
        batch_index: torch.Tensor,
        cont_covs=None,
        cat_covs=None,
        size_factor=None,
        y=None,
        transform_batch=None,
    ):
        if cont_covs is None:
            decoder_input = torch.cat([z, u], dim=-1)
        elif z.dim() != cont_covs.dim():
            decoder_input = torch.cat(
                [z, u, cont_covs.unsqueeze(0).expand(z.size(0), -1, -1)], dim=-1
            )
        else:
            decoder_input = torch.cat([z, u, cont_covs], dim=-1)

        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch

        if not self.use_size_factor_key:
            size_factor = library

        px_scale, px_r, px_rate, px_dropout = self.x_decoder(
            self.dispersion,
            decoder_input,
            size_factor,
            batch_index,
            *categorical_input,
            y,
        )

        if self.dispersion == "gene-label":
            px_r = F.linear(
                one_hot(y, self.n_labels), self.px_r
            )
        elif self.dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
        elif self.dispersion == "gene":
            px_r = self.px_r

        px_r = torch.exp(px_r)
        
        ps_m, ps_v = self.s_decoder(u)
        if self.b_decoder is not None:
            pb_m, pb_v = self.b_decoder(z)
        else:
            pb_m, pb_v = None, None

        return dict(
            px_scale=px_scale, px_r=px_r, px_rate=px_rate, px_dropout=px_dropout,
            ps_m=ps_m, ps_v=ps_v, pb_m=pb_m, pb_v=pb_v
        )


    @staticmethod
    def reconstruction_loss(
        x: torch.Tensor,
        px_rate: torch.Tensor,
        px_r: torch.Tensor,
        px_dropout: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute likelihood loss for zero-inflated negative binomial distribution.

        Args:
        ----
            x: Input data.
            px_rate: Mean of distribution.
            px_r: Inverse dispersion.
            px_dropout: Logits scale of zero inflation probability.

        Returns
        -------
            Negative log likelihood (reconstruction loss) for each data point. If number
            of latent samples == 1, the tensor has shape `(batch_size, )`. If number
            of latent samples > 1, the tensor has shape `(n_samples, batch_size)`.
        """
        recon_loss = (
            -ZeroInflatedNegativeBinomial(mu=px_rate, theta=px_r, zi_logits=px_dropout)
            .log_prob(x)
            .sum(dim=-1)
        )
        return recon_loss

    def pheno_reconstruction_loss(
        self,
        categorical_s: torch.Tensor,
        continuous_s: torch.Tensor,
        ps_m: torch.Tensor,
        ps_v: torch.Tensor,
    ):
        """
        Compute likelihood loss for zero-inflated negative binomial distribution.

        Args:
        ----
            s: Input phenotypes.
            ps_m: Mean of distribution.

        Returns
        -------
            Negative log likelihood (reconstruction loss) for each data point. If number
            of latent samples == 1, the tensor has shape `(batch_size, )`. If number
            of latent samples > 1, the tensor has shape `(n_samples, batch_size)`.
        """
        categorical_pheno_recon_loss = 0
        continuous_pheno_recon_loss = 0
        self.pheno_continuous_recon_penalty = self.pheno_continuous_recon_penalty.to(ps_m.device)
        self.pheno_categorical_recon_penalty = self.pheno_categorical_recon_penalty.to(ps_m.device)
        continuous_pheno_recon_loss_list = []
        if self.n_continuous_pheno > 0:
            for i in range(self.n_continuous_pheno):
                labelled_idx = torch.where(continuous_s[:, i] != -np.inf)[0]
                ps_m_temp = ps_m[labelled_idx, i].unsqueeze(1)
                continuous_s_temp = continuous_s[labelled_idx, i].unsqueeze(1)
                continuous_pheno_recon_loss_temp = torch.sum(
                    MSELoss(reduce=False)(
                    ps_m_temp, continuous_s_temp), dim=1) * self.pheno_continuous_recon_penalty[i]
                continuous_pheno_recon_loss += continuous_pheno_recon_loss_temp
                continuous_pheno_recon_loss_list.append(torch.mean(continuous_pheno_recon_loss_temp))
                
        categorical_pheno_recon_loss_list = []
        if self.n_categorical_pheno > 0:
            idx = self.n_continuous_pheno
            for i in range(self.n_categorical_pheno):
                categorical_pheno_recon_loss_temp = CrossEntropyLoss(reduce=False)(
                    ps_m[:, idx:(idx+self.n_categorical_per_pheno[i])], categorical_s[:, i]
                ) * self.pheno_categorical_recon_penalty[i]
                categorical_pheno_recon_loss += categorical_pheno_recon_loss_temp
                categorical_pheno_recon_loss_list.append(torch.mean(categorical_pheno_recon_loss_temp))
                idx += self.n_categorical_per_pheno[i]
        return continuous_pheno_recon_loss + categorical_pheno_recon_loss

    def back_reconstruction_loss(
        self,
        categorical_b: torch.Tensor,
        continuous_b: torch.Tensor,
        pb_m: torch.Tensor,
        pb_v: torch.Tensor,
    ):
        """
        Compute likelihood loss for zero-inflated negative binomial distribution.

        Args:
        ----
            s: Input phenotypes.
            ps_m: Mean of distribution.

        Returns
        -------
            Negative log likelihood (reconstruction loss) for each data point. If number
            of latent samples == 1, the tensor has shape `(batch_size, )`. If number
            of latent samples > 1, the tensor has shape `(n_samples, batch_size)`.
        """
        categorical_back_recon_loss = 0
        continuous_back_recon_loss = 0
        self.back_continuous_recon_penalty = self.back_continuous_recon_penalty.to(pb_m.device)
        self.back_categorical_recon_penalty = self.back_categorical_recon_penalty.to(pb_m.device)
        continuous_back_recon_loss_list = []
        if self.n_continuous_back > 0:
            for i in range(self.n_continuous_back):
                continuous_back_recon_loss_temp = torch.sum(
                    MSELoss(reduce=False)(
                    pb_m, continuous_b), dim=1) * self.back_continuous_recon_penalty
                continuous_back_recon_loss += continuous_back_recon_loss_temp
                continuous_back_recon_loss_list.append(torch.mean(continuous_back_recon_loss_temp))

        categorical_back_recon_loss_list = []
        if self.n_categorical_back > 0:
            idx = self.n_continuous_back
            for i in range(self.n_categorical_back):
                categorical_back_recon_loss_temp = CrossEntropyLoss(reduce=False)(
                    pb_m[:, idx:(idx+self.n_categorical_per_back[i])], categorical_b[:, i]
                ) * self.back_categorical_recon_penalty[i]
                categorical_back_recon_loss += categorical_back_recon_loss_temp
                categorical_back_recon_loss_list.append(torch.mean(categorical_back_recon_loss_temp))
                idx += self.n_categorical_per_back[i]
        return continuous_back_recon_loss + categorical_back_recon_loss

    @staticmethod
    def latent_kl_divergence(
        variational_mean: torch.Tensor,
        variational_var: torch.Tensor,
        prior_mean: torch.Tensor,
        prior_var: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KL divergence between a variational posterior and prior Gaussian.
        Args:
        ----
            variational_mean: Mean of the variational posterior Gaussian.
            variational_var: Variance of the variational posterior Gaussian.
            prior_mean: Mean of the prior Gaussian.
            prior_var: Variance of the prior Gaussian.

        Returns
        -------
            KL divergence for each data point. If number of latent samples == 1,
            the tensor has shape `(batch_size, )`. If number of latent
            samples > 1, the tensor has shape `(n_samples, batch_size)`.
        """
        return kl(
            Normal(variational_mean, variational_var.sqrt()),
            Normal(prior_mean, prior_var.sqrt()),
        ).sum(dim=-1)

    def library_kl_divergence(
        self,
        batch_index: torch.Tensor,
        variational_library_mean: torch.Tensor,
        variational_library_var: torch.Tensor,
        library: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KL divergence between library size variational posterior and prior.
        Both the variational posterior and prior are Normal.
        Args:
        ----
            batch_index: Batch indices for batch-specific library size mean and
                variance.
            variational_library_mean: Mean of variational Normal.
            variational_library_var: Variance of variational Normal.
            library: Sampled library size.
        Returns
        -------
            KL divergence for each data point. If number of latent samples == 1,
            the tensor has shape `(batch_size, )`. If number of latent
            samples > 1, the tensor has shape `(n_samples, batch_size)`.
        """
        if not self.use_observed_lib_size:
            (
                local_library_log_means,
                local_library_log_vars,
            ) = self._compute_local_library_params(batch_index)

            kl_library = kl(
                Normal(variational_library_mean, variational_library_var.sqrt()),
                Normal(local_library_log_means, local_library_log_vars.sqrt()),
            )
        else:
            kl_library = torch.zeros_like(library)
        return kl_library.sum(dim=-1)

    def _generic_loss(
        self,
        tensors: torch.Tensor,
        inference_outputs: Dict[str, torch.Tensor],
        generative_outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        x = tensors[REGISTRY_KEYS.X_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]

        categorical_s = None
        continuous_s = None
        if self.n_categorical_pheno > 0:
            categorical_s = tensors[REGISTRY_KEYS.CAT_PHENOS_KEY].to(torch.long)
        if self.n_continuous_pheno > 0:
            continuous_s = tensors[REGISTRY_KEYS.CONT_PHENOS_KEY]
        
        categorical_b = None
        continuous_b = None
        if self.n_categorical_back > 0:
            categorical_b = tensors[REGISTRY_KEYS.CAT_BACKS_KEY].to(torch.long)
        if self.n_continuous_back > 0:
            continuous_b = tensors[REGISTRY_KEYS.CONT_BACKS_KEY]

        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        qu_m = inference_outputs["qu_m"]
        qu_v = inference_outputs["qu_v"]
        library = inference_outputs["library"]
        ql_m = inference_outputs["ql_m"]
        ql_v = inference_outputs["ql_v"]
        px_rate = generative_outputs["px_rate"]
        px_r = generative_outputs["px_r"]
        px_dropout = generative_outputs["px_dropout"]
        ps_m = generative_outputs["ps_m"]
        ps_v = generative_outputs["ps_v"]
        pb_m = generative_outputs["pb_m"]
        pb_v = generative_outputs["pb_v"]

        prior_z_m = torch.zeros_like(qz_m)
        prior_z_v = torch.ones_like(qz_v)
        prior_u_m = torch.zeros_like(qu_m)
        prior_u_v = torch.ones_like(qu_v)

        recon_loss = self.reconstruction_loss(x, px_rate, px_r, px_dropout)
        pheno_recon_loss = self.pheno_reconstruction_loss(categorical_s, continuous_s, ps_m, ps_v)
        if (pb_m is not None) & (pb_v is not None):
            back_recon_loss = self.back_reconstruction_loss(categorical_b, continuous_b, pb_m, pb_v)
        else:
            back_recon_loss = torch.tensor(0.0)
        kl_z = self.latent_kl_divergence(qz_m, qz_v, prior_z_m, prior_z_v)
        kl_u = self.latent_kl_divergence(qu_m, qu_v, prior_u_m, prior_u_v)
        kl_library = self.library_kl_divergence(batch_index, ql_m, ql_v, library)
        return dict(
            recon_loss=recon_loss,
            pheno_recon_loss=pheno_recon_loss,
            back_recon_loss=back_recon_loss,
            kl_z=kl_z,
            kl_u=kl_u,
            kl_library=kl_library,
        )

    def loss(
        self,
        tensors,
        inference_outputs: Dict[str, torch.Tensor],
        generative_outputs: Dict[str, torch.Tensor],
        kl_weight: float = 1.0,
    ) -> LossRecorder:
        """
        Compute loss terms for ACE.
        Args:
        ----
            tensors: 
            inference_outputs: Dictionary of inference step outputs. The keys
                are "background" and "target" for the corresponding outputs.
            generative_outputs: Dictionary of generative step outputs. The keys
                are "background" and "target" for the corresponding outputs.
            kl_weight: Importance weight for KL divergence of background and salient
                latent variables, relative to KL divergence of library size.

        Returns
        -------
            An scvi.module.base.LossRecorder instance that records the following:
            loss: One-dimensional tensor for overall loss used for optimization.
            reconstruction_loss: Reconstruction loss with shape
                `(n_samples, batch_size)` if number of latent samples > 1, or
                `(batch_size, )` if number of latent samples == 1.
            kl_local: KL divergence term with shape
                `(n_samples, batch_size)` if number of latent samples > 1, or
                `(batch_size, )` if number of latent samples == 1.
            kl_global: One-dimensional tensor for global KL divergence term.
        """
        losses = self._generic_loss(
            tensors,
            inference_outputs,
            generative_outputs,
        )
        reconst_loss = losses["recon_loss"]
        pheno_recon_loss = losses["pheno_recon_loss"]
        back_recon_loss = losses["back_recon_loss"]

        kl_divergence_z = losses["kl_z"]
        kl_divergence_u = losses["kl_u"]
        kl_divergence_l = losses["kl_library"]


        kl_local_for_warmup = kl_divergence_z + kl_divergence_u
        kl_local_no_warmup = kl_divergence_l

        kl_local_loss = (
            kl_weight
            * (kl_local_for_warmup)
            + kl_local_no_warmup
        )

        hsic_loss = hsic(inference_outputs['z'], inference_outputs['u'])

        continuous_s = tensors[REGISTRY_KEYS.CONT_PHENOS_KEY]
        if -np.inf in continuous_s:
            loss = torch.mean(reconst_loss + kl_local_loss + back_recon_loss + 
            self.hsic_loss_penalty * hsic_loss) + torch.mean(pheno_recon_loss)
        else:
            loss = torch.mean(reconst_loss + kl_local_loss + pheno_recon_loss + back_recon_loss + 
            self.hsic_loss_penalty * hsic_loss)
        kl_local = dict(
            kl_divergence_l=kl_divergence_l,
            kl_divergence_z=kl_divergence_z,
            kl_divergence_u=kl_divergence_u,
        )
        kl_global = torch.tensor(0.0)

        return LossRecorder(
            loss,
            reconst_loss,
            kl_local,
            kl_global,
            pheno_recon_loss=torch.mean(pheno_recon_loss),
            back_recon_loss=torch.mean(back_recon_loss),
            hsic_loss=hsic_loss ,
        )

    @torch.no_grad()
    def sample(self):
        raise NotImplementedError

    @torch.no_grad()
    @auto_move_data
    def marginal_ll(self):
        raise NotImplementedError
