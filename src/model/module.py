"""MERN module for inferring metabolic state from single-cell transcriptomic data."""

from typing import Callable, Dict, Optional, Tuple, Literal
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn.functional import one_hot

from scvi import REGISTRY_KEYS
from scvi.module.base import BaseModuleClass, LossOutput
from scvi.module.base import (
    BaseMinifiedModeModuleClass,
    EmbeddingModuleMixin,
    LossOutput,
    auto_move_data,
)

from torch.distributions import Distribution
from .constants import MODULE_KEYS, GRAPH_REGISTRY_KEYS, METABOLIC_REGISTRY_KEYS

class MERNModule(BaseModuleClass):
    """MERN module for inferring metabolic state from single-cell transcriptomic data.
    
    This module implements the core neural network architecture for MERN.

    Parameters
    ----------
    genes: list[str]
        List of genes in order 
    vertices: dict[int, str]
        Dictionary mapping indices to vertices
    rxn_to_genes
        Dictionary mapping reactions to genes
    n_input: int
        Number of input features
    n_batch: int
        Number of batches. If ``0``, no batch correction is performed.
    n_labels: int
        Number of labels
    n_hidden: int
        Number of nodes per hidden layer
    n_metabolic_dim: int
        Dimensionality of the metabolic latent space
    n_background_dim: int
        Dimensionality of the background latent space
    n_layers: int
        Number of hidden layers
    dropout_rate: float
        Dropout rate
    n_continuous_cov
        Number of continuous covariates.
    n_cats_per_cov
        A list of integers containing the number of categories for each categorical covariate.
    dropout_rate
        Dropout rate. Passed into :class:`~scvi.nn.Encoder` but not :class:`~scvi.nn.DecoderSCVI`.
    dispersion
        Flexibility of the dispersion parameter when ``gene_likelihood`` is either ``"nb"`` or
        ``"zinb"``. One of the following:

        * ``"gene"``: parameter is constant per gene across cells.
        * ``"gene-batch"``: parameter is constant per gene per batch.
        * ``"gene-label"``: parameter is constant per gene per label.
        * ``"gene-cell"``: parameter is constant per gene per cell.
    log_variational
        If ``True``, use :func:`~torch.log1p` on input data before encoding for numerical stability
        (not normalization).
    gene_likelihood
        Distribution to use for reconstruction in the generative process. One of the following:

        * ``"nb"``: :class:`~scvi.distributions.NegativeBinomial`.
        * ``"zinb"``: :class:`~scvi.distributions.ZeroInflatedNegativeBinomial`.
        * ``"poisson"``: :class:`~scvi.distributions.Poisson`.
        * ``"normal"``: :class:`~torch.distributions.Normal`.
    latent_distribution
        Distribution to use for the latent space. One of the following:

        * ``"normal"``: isotropic normal.
        * ``"ln"``: logistic normal with normal params N(0, 1).
    encode_covariates
        If ``True``, covariates are concatenated to gene expression prior to passing through
        the encoder(s). Else, only gene expression is used.
    deeply_inject_covariates
        If ``True`` and ``n_layers > 1``, covariates are concatenated to the outputs of hidden
        layers in the encoder(s) (if ``encoder_covariates`` is ``True``) and the decoder prior to
        passing through the next layer.
    batch_representation
        ``EXPERIMENTAL`` Method for encoding batch information. One of the following:

        * ``"one-hot"``: represent batches with one-hot encodings.
        * ``"embedding"``: represent batches with continuously-valued embeddings using
          :class:`~scvi.nn.Embedding`.

        Note that batch representations are only passed into the encoder(s) if
        ``encode_covariates`` is ``True``.
    use_batch_norm
        Specifies where to use :class:`~torch.nn.BatchNorm1d` in the model. One of the following:

        * ``"none"``: don't use batch norm in either encoder(s) or decoder.
        * ``"encoder"``: use batch norm only in the encoder(s).
        * ``"decoder"``: use batch norm only in the decoder.
        * ``"both"``: use batch norm in both encoder(s) and decoder.

        Note: if ``use_layer_norm`` is also specified, both will be applied (first
        :class:`~torch.nn.BatchNorm1d`, then :class:`~torch.nn.LayerNorm`).
    use_layer_norm
        Specifies where to use :class:`~torch.nn.LayerNorm` in the model. One of the following:

        * ``"none"``: don't use layer norm in either encoder(s) or decoder.
        * ``"encoder"``: use layer norm only in the encoder(s).
        * ``"decoder"``: use layer norm only in the decoder.
        * ``"both"``: use layer norm in both encoder(s) and decoder.

        Note: if ``use_batch_norm`` is also specified, both will be applied (first
        :class:`~torch.nn.BatchNorm1d`, then :class:`~torch.nn.LayerNorm`).
    use_size_factor_key
        If ``True``, use the :attr:`~anndata.AnnData.obs` column as defined by the
        ``size_factor_key`` parameter in the model's ``setup_anndata`` method as the scaling
        factor in the mean of the conditional distribution. Takes priority over
        ``use_observed_lib_size``.
    use_observed_lib_size
        If ``True``, use the observed library size for RNA as the scaling factor in the mean of the
        conditional distribution.
    extra_payload_autotune
        If ``True``, will return extra matrices in the loss output to be used during autotune
    library_log_means
        :class:`~numpy.ndarray` of shape ``(1, n_batch)`` of means of the log library sizes that
        parameterize the prior on library size if ``use_size_factor_key`` is ``False`` and
        ``use_observed_lib_size`` is ``False``.
    library_log_vars
        :class:`~numpy.ndarray` of shape ``(1, n_batch)`` of variances of the log library sizes
        that parameterize the prior on library size if ``use_size_factor_key`` is ``False`` and
        ``use_observed_lib_size`` is ``False``.
    var_activation
        Callable used to ensure positivity of the variance of the variational distribution. Passed
        into :class:`~scvi.nn.Encoder`. Defaults to :func:`~torch.exp`.
    extra_encoder_kwargs
        Additional keyword arguments passed into :class:`~scvi.nn.Encoder`.
    extra_decoder_kwargs
        Additional keyword arguments passed into :class:`~scvi.nn.DecoderSCVI`.
    batch_embedding_kwargs
        Keyword arguments passed into :class:`~scvi.nn.Embedding` if ``batch_representation`` is
        set to ``"embedding"``.
    
    """
    
    def __init__(
        self,
        genes: list[str],
        vertices: dict[int, str],
        rxn_to_genes: dict[str, list[str]],
        metabolic_genes: pd.Series,
        n_input: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_metabolic_dim: int = 10,
        n_background_dim: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        n_continuous_cov: int = 0,
        n_cats_per_cov: list[int] | None = None,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        log_variational: bool = True,
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
        latent_distribution: Literal["normal", "ln"] = "normal",
        encode_covariates: bool = False,
        deeply_inject_covariates: bool = True,
        batch_representation: Literal["one-hot", "embedding"] = "one-hot",
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_size_factor_key: bool = False,
        use_observed_lib_size: bool = True,
        extra_payload_autotune: bool = False,
        library_log_means_met: np.ndarray | None = None,
        library_log_vars_met: np.ndarray | None = None,
        library_log_means_back: np.ndarray | None = None,
        library_log_vars_back: np.ndarray | None = None,
        var_activation: Callable[[torch.Tensor], torch.Tensor] = None,
        extra_encoder_kwargs: dict | None = None,
        extra_decoder_kwargs: dict | None = None,
        batch_embedding_kwargs: dict | None = None,
        positive_met_dims: bool = False,
        fixed_rxn_genes: bool = False,
        strict_met_back_separation: bool = False,
        rxn_genes_bias: bool = False,
        fixed_graph_cell_kl: bool = False,
    ):
        from scvi.nn import Encoder
        from .base_components import (
            DecoderMERN,
            GraphDecoder,
            GraphEncoder,
            MetabolicEncoder,
        )

        super().__init__()
        self.genes = genes
        self.metabolic_genes = metabolic_genes
        self.n_input = n_input
        self.n_batch = n_batch
        self.n_hidden = n_hidden
        self.n_metabolic_dim = n_metabolic_dim
        self.n_background_dim = n_background_dim
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.vertices = vertices
        self.rxn_to_genes = rxn_to_genes
        self.n_continuous_cov = n_continuous_cov
        self.n_cats_per_cov = n_cats_per_cov
        self.dispersion = dispersion
        self.log_variational = log_variational
        self.gene_likelihood = gene_likelihood
        self.latent_distribution = latent_distribution
        self.encode_covariates = encode_covariates
        self.use_size_factor_key = use_size_factor_key
        self.use_observed_lib_size = use_size_factor_key or use_observed_lib_size
        self.extra_payload_autotune = extra_payload_autotune
        self.strict_met_back_separation = strict_met_back_separation
        self.fixed_graph_cell_kl = fixed_graph_cell_kl
        self.positive_met_dims = positive_met_dims

        # save gene order
        self.gene_is_metabolic = []
        for gene in genes:
            self.gene_is_metabolic.append(True if metabolic_genes[gene] == "Metabolic" else False)
        self.gene_is_metabolic = np.array(self.gene_is_metabolic)

        if not self.use_observed_lib_size:
            if library_log_means_met is None or library_log_vars_met is None or library_log_means_back is None or library_log_vars_back is None:
                raise ValueError(
                    "If not using observed_lib_size, "
                    "must provide library_log_means and library_log_vars."
                )

            self.register_buffer("library_log_means_met", torch.from_numpy(library_log_means_met).float())
            self.register_buffer("library_log_vars_met", torch.from_numpy(library_log_vars_met).float())
            self.register_buffer("library_log_means_back", torch.from_numpy(library_log_means_back).float())
            self.register_buffer("library_log_vars_back", torch.from_numpy(library_log_vars_back).float())

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
                "`dispersion` must be one of 'gene', 'gene-batch', 'gene-label', 'gene-cell'."
            )

        self.batch_representation = batch_representation
        if self.batch_representation == "embedding":
            self.init_embedding(REGISTRY_KEYS.BATCH_KEY, n_batch, **(batch_embedding_kwargs or {}))
            batch_dim = self.get_embedding(REGISTRY_KEYS.BATCH_KEY).embedding_dim
        elif self.batch_representation != "one-hot":
            raise ValueError("`batch_representation` must be one of 'one-hot', 'embedding'.")
        
        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        n_input_encoder = n_input + n_continuous_cov * encode_covariates
        if self.batch_representation == "embedding":
            n_input_encoder += batch_dim * encode_covariates
            cat_list = list([] if n_cats_per_cov is None else n_cats_per_cov)
        else:
            cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)

        encoder_cat_list = cat_list if encode_covariates else None
        _extra_encoder_kwargs = extra_encoder_kwargs or {}
        self.m_encoder = MetabolicEncoder(
            n_input_encoder,
            n_metabolic_dim,
            n_cat_list=encoder_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            return_dist=True,
            positive_met_dims=positive_met_dims,
            **_extra_encoder_kwargs,
        )
        self.b_encoder = Encoder(
            n_input_encoder,
            n_background_dim,
            n_cat_list=encoder_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            return_dist=True,
            **_extra_encoder_kwargs,
        )
        # l encoder goes from n_input-dimensional data to 1-d library size, for metabolic library size
        self.lm_encoder = Encoder(
            n_input_encoder,
            1,
            n_layers=1,
            n_cat_list=encoder_cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            return_dist=True,
            **_extra_encoder_kwargs,
        )
        # l encoder goes from n_input-dimensional data to 1-d library size, for background library size
        self.lb_encoder = Encoder(
            n_input_encoder,
            1,
            n_layers=1,
            n_cat_list=encoder_cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            return_dist=True,
            **_extra_encoder_kwargs,
        )

        self.v_encoder = GraphEncoder(
            vnum=len(vertices),
            n_output=n_metabolic_dim,
            return_dist=True,
            positive_met_dims=positive_met_dims,
        )

        n_input_metabolic_decoder = n_metabolic_dim + n_continuous_cov
        n_input_background_decoder = n_background_dim + n_continuous_cov

        if self.batch_representation == "embedding":
            n_input_metabolic_decoder += batch_dim
            n_input_background_decoder += batch_dim

        _extra_decoder_kwargs = extra_decoder_kwargs or {}
        self.decoder = DecoderMERN(
            genes=genes,
            rxns=vertices,
            background_n_input=n_input_background_decoder,
            rxns_to_genes=rxn_to_genes,
            metabolic_genes=metabolic_genes,
            n_cat_list=cat_list,
            fixed_rxn_genes=fixed_rxn_genes,
            strict_met_back_separation=strict_met_back_separation,
            rxn_genes_bias=rxn_genes_bias,
        )

        self.graph_decoder = GraphDecoder()
        

    def _get_inference_input(
            self, 
            tensors: tuple[dict[str, torch.Tensor | None], dict[str, torch.Tensor | None]],
    ) -> dict[str, torch.Tensor | None]:
        """Get input tensors for the generative process"""
        return {
            MODULE_KEYS.X_KEY: tensors['cells'][REGISTRY_KEYS.X_KEY],
            MODULE_KEYS.BATCH_INDEX_KEY: tensors['cells'][REGISTRY_KEYS.BATCH_KEY],
            MODULE_KEYS.CONT_COVS_KEY: tensors['cells'].get(REGISTRY_KEYS.CONT_COVS_KEY, None),
            MODULE_KEYS.CAT_COVS_KEY: tensors['cells'].get(REGISTRY_KEYS.CAT_COVS_KEY, None),
            MODULE_KEYS.EIDX_KEY: tensors['graph'][GRAPH_REGISTRY_KEYS.EIDX_KEY],
            MODULE_KEYS.EWT_KEY: tensors['graph'][GRAPH_REGISTRY_KEYS.EWT_KEY],
            MODULE_KEYS.ESGN_KEY: tensors['graph'][GRAPH_REGISTRY_KEYS.ESGN_KEY],
        }
    
    def _get_generative_input(
            self, 
            tensors: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
            inference_outputs: dict[str, torch.Tensor | Distribution | None],
    ) -> dict[str, torch.Tensor | None]:
        """Get input tensors for the generative process"""
        met_size_factor = tensors['cells'].get(METABOLIC_REGISTRY_KEYS.MET_SIZE_FACTOR_KEY, None)
        back_size_factor = tensors['cells'].get(METABOLIC_REGISTRY_KEYS.BACK_SIZE_FACTOR_KEY, None)

        if met_size_factor is not None:
            met_size_factor = torch.log(met_size_factor)
        if back_size_factor is not None:
            back_size_factor = torch.log(back_size_factor)

        return {
            MODULE_KEYS.M_KEY: inference_outputs[MODULE_KEYS.M_KEY],
            MODULE_KEYS.B_KEY: inference_outputs[MODULE_KEYS.B_KEY],
            MODULE_KEYS.V_KEY: inference_outputs[MODULE_KEYS.V_KEY],
            MODULE_KEYS.EIDX_KEY: tensors['graph'][GRAPH_REGISTRY_KEYS.EIDX_KEY],
            MODULE_KEYS.EWT_KEY: tensors['graph'][GRAPH_REGISTRY_KEYS.EWT_KEY],
            MODULE_KEYS.ESGN_KEY: tensors['graph'][GRAPH_REGISTRY_KEYS.ESGN_KEY],
            MODULE_KEYS.METABOLIC_LIBRARY_KEY: inference_outputs[MODULE_KEYS.METABOLIC_LIBRARY_KEY],
            MODULE_KEYS.BACKGROUND_LIBRARY_KEY: inference_outputs[MODULE_KEYS.BACKGROUND_LIBRARY_KEY],
            MODULE_KEYS.BATCH_INDEX_KEY: tensors['cells'][REGISTRY_KEYS.BATCH_KEY],
            MODULE_KEYS.CONT_COVS_KEY: tensors['cells'].get(REGISTRY_KEYS.CONT_COVS_KEY, None),
            MODULE_KEYS.CAT_COVS_KEY: tensors['cells'].get(REGISTRY_KEYS.CAT_COVS_KEY, None),
            MODULE_KEYS.MET_SIZE_FACTOR_KEY: met_size_factor,
            MODULE_KEYS.BACK_SIZE_FACTOR_KEY: back_size_factor,
            MODULE_KEYS.Y_KEY: tensors['cells'][REGISTRY_KEYS.LABELS_KEY],
        }
    
    def _compute_local_library_params(
        self,
        batch_index: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        """Computes local library parameters.

        Compute tuple of two tensors of shape (batch_index.shape[0], 1) where each
        element corresponds to the mean and variances, respectively, of the
        log library sizes in the batch the cell corresponds to.
        """
        from torch.nn.functional import linear

        n_batch = self.library_log_means_met.shape[1]
        local_library_log_means_met = linear(
            one_hot(batch_index.squeeze(-1), n_batch).float(), self.library_log_means_met
        )

        local_library_log_vars_met = linear(
            one_hot(batch_index.squeeze(-1), n_batch).float(), self.library_log_vars
        )

        local_library_log_means_back = linear(
            one_hot(batch_index.squeeze(-1), n_batch).float(), self.library_log_means_back
        )

        local_library_log_vars_back = linear(
            one_hot(batch_index.squeeze(-1), n_batch).float(), self.library_log_vars_back
        )

        return (local_library_log_means_met, local_library_log_vars_met), (local_library_log_means_back, local_library_log_vars_back)
    
    @auto_move_data
    def inference(
        self,
        x: torch.Tensor,
        batch_index: torch.Tensor,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        eidx: torch.Tensor | None = None,
        ewt: torch.Tensor | None = None,
        esgn: torch.Tensor | None = None,
        n_samples: int = 1,
    ) -> dict[str, torch.Tensor | Distribution | None]:
        """Run the inference process."""
        x_ = x
        if self.use_observed_lib_size:
            met_library = torch.log(x[:, self.gene_is_metabolic].sum(1)).unsqueeze(1)
            back_library = torch.log(x[:, ~self.gene_is_metabolic].sum(1)).unsqueeze(1)
        if self.log_variational:
            x_ = torch.log1p(x_)

        if cont_covs is not None and self.encode_covariates:
            encoder_input = torch.cat((x_, cont_covs), dim=-1)
        else:
            encoder_input = x_
        if cat_covs is not None and self.encode_covariates:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()

        if self.batch_representation == "embedding" and self.encode_covariates:
            batch_rep = self.compute_embedding(REGISTRY_KEYS.BATCH_KEY, batch_index)
            encoder_input = torch.cat([encoder_input, batch_rep], dim=-1)
            qm, m = self.m_encoder(encoder_input, *categorical_input)
            qb, b = self.b_encoder(encoder_input, *categorical_input)
        else:
            qm, m = self.m_encoder(encoder_input, batch_index, *categorical_input)
            qb, b = self.b_encoder(encoder_input, batch_index, *categorical_input)

        # graph encoding
        qv, v = self.v_encoder(eidx, ewt, esgn)

        qlm, qlb = None, None
        if not self.use_observed_lib_size:
            if self.batch_representation == "embedding":
                qlm, met_library_encoded = self.lm_encoder(encoder_input, *categorical_input)
                qlb, back_library_encoded = self.lb_encoder(encoder_input, *categorical_input)
            else:
                qlm, met_library_encoded = self.lm_encoder(
                    encoder_input, batch_index, *categorical_input
                )
                qlb, back_library_encoded = self.lb_encoder(
                    encoder_input, batch_index, *categorical_input
                )
            met_library = met_library_encoded
            back_library = back_library_encoded
        
        if n_samples > 1:
            untran_m = qm.sample((n_samples,))
            untran_b = qb.sample((n_samples,))
            m = self.m_encoder.z_transformation(untran_m)
            b = self.b_encoder.z_transformation(untran_b)
            untran_v = qv.sample((n_samples,))
            v = self.v_encoder.z_transformation(untran_v)
            if self.use_observed_lib_size:
                met_library = met_library.unsqueeze(0).expand(
                    (n_samples, met_library.size(0), met_library.size(1))
                )
                back_library = back_library.unsqueeze(0).expand(
                    (n_samples, back_library.size(0), back_library.size(1))
                )
            else:
                met_library = qlm.sample((n_samples,))
                back_library = qlb.sample((n_samples,))

        return {
            MODULE_KEYS.M_KEY: m,
            MODULE_KEYS.B_KEY: b,
            MODULE_KEYS.V_KEY: v,
            MODULE_KEYS.QM_KEY: qm,
            MODULE_KEYS.QB_KEY: qb,
            MODULE_KEYS.QV_KEY: qv,
            MODULE_KEYS.QLM_KEY: qlm,
            MODULE_KEYS.QLB_KEY: qlb,
            MODULE_KEYS.METABOLIC_LIBRARY_KEY: met_library,
            MODULE_KEYS.BACKGROUND_LIBRARY_KEY: back_library,
        }
    
    @auto_move_data
    def generative(
        self,
        m: torch.Tensor,
        b: torch.Tensor,
        v: torch.Tensor,
        eidx: torch.Tensor,
        ewt: torch.Tensor,
        esgn: torch.Tensor,
        metabolic_library: torch.Tensor,
        background_library: torch.Tensor,
        batch_index: torch.Tensor,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        met_size_factor: torch.Tensor | None = None,
        back_size_factor: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        transform_batch: torch.Tensor | None = None,
    ) -> dict[str, Distribution | None]:
        """Run the generative process."""
        from torch.nn.functional import linear, one_hot

        from scvi.distributions import (
            NegativeBinomial,
            Normal,
        )

        from torch.distributions import Bernoulli

        if cont_covs is None:
            decoder_input_m = m
            decoder_input_b = b
        elif m.dim() != cont_covs.dim():
            decoder_input_m = torch.cat(
                [m, cont_covs.unsqueeze(0).expand(m.size(0), -1, -1)], dim=-1
            )
        elif b.dim() != cont_covs.dim():
            decoder_input_b = torch.cat(
                [b, cont_covs.unsqueeze(0).expand(b.size(0), -1, -1)], dim=-1
            )
        else:
            decoder_input_m = torch.cat([m, cont_covs], dim=-1)
            decoder_input_b = torch.cat([b, cont_covs], dim=-1)

        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch

        if not self.use_size_factor_key:
            met_size_factor = metabolic_library
            back_size_factor = background_library
        
        if self.batch_representation == "embedding":
            batch_rep = self.compute_embedding(REGISTRY_KEYS.BATCH_KEY, batch_index)
            decoder_input_m = torch.cat([decoder_input_m, batch_rep], dim=-1)
            decoder_input_b = torch.cat([decoder_input_b, batch_rep], dim=-1)
            decode_output = self.decoder(
                self.dispersion,
                decoder_input_m,
                decoder_input_b,
                v,
                met_size_factor,
                back_size_factor,
                *categorical_input,
            )
            px_met_scale, px_met_r, px_met_rate = decode_output[0]
            px_back_scale, px_back_r, px_back_rate = decode_output[1]
            enzyme_activity = decode_output[2]
        else:
            decode_output = self.decoder(
                self.dispersion,
                decoder_input_m,
                decoder_input_b,
                v,
                met_size_factor,
                back_size_factor,
                batch_index,
                *categorical_input,
            )
            px_met_scale, px_met_r, px_met_rate = decode_output[0]
            px_back_scale, px_back_r, px_back_rate = decode_output[1]
            enzyme_activity = decode_output[2]
        if self.dispersion == "gene-label":
            px_r = linear(
                one_hot(y.squeeze(-1), self.n_labels).float(), self.px_r
            )  # px_r gets transposed - last dimension is nb genes
        elif self.dispersion == "gene-batch":
            px_r = linear(one_hot(batch_index.squeeze(-1), self.n_batch).float(), self.px_r)
        elif self.dispersion == "gene":
            px_r = self.px_r
        
        px_r = torch.exp(px_r)

        if self.gene_likelihood == "nb":
            px_met = NegativeBinomial(mu=px_met_rate, theta=px_r, scale=px_met_scale)
            px_back = NegativeBinomial(mu=px_back_rate, theta=px_r, scale=px_back_scale)
            # not sure the best way to do scale here, maybe renormalize, or add met and back scales?
            px_scale = nn.functional.softmax(px_met_rate + px_back_rate, dim=1)
            # this is weird for px_scale
            px = NegativeBinomial(mu=px_met_rate + px_back_rate, theta=px_r, scale=px_scale)

        g_logits = self.graph_decoder(v, eidx, esgn)
        pg = Bernoulli(logits=g_logits)
        
        # Priors
        if self.use_observed_lib_size:
            pl_met = None
            pl_back = None
        else:
            met_tuple, back_tuple = self._compute_local_library_params(batch_index)
            pl_met = Normal(met_tuple[0], met_tuple[1].sqrt())
            pl_back = Normal(back_tuple[0], back_tuple[1].sqrt())
        
        if self.positive_met_dims:
            pm = Normal(torch.ones_like(m)*5, torch.ones_like(m))
            pv = Normal(torch.ones_like(v)*5, torch.ones_like(v))
        else:
            pm = Normal(torch.zeros_like(m), torch.ones_like(m))
            pv = Normal(torch.zeros_like(v), torch.ones_like(v))
        pb = Normal(torch.zeros_like(b), torch.ones_like(b))
        

        return {
            MODULE_KEYS.PX_KEY: px,
            MODULE_KEYS.PM_KEY: pm,
            MODULE_KEYS.PB_KEY: pb,
            MODULE_KEYS.PV_KEY: pv,
            MODULE_KEYS.PLM_KEY: pl_met,
            MODULE_KEYS.PLB_KEY: pl_back,
            MODULE_KEYS.PX_MET_KEY: px_met,
            MODULE_KEYS.PX_BACK_KEY: px_back,
            MODULE_KEYS.PG_KEY: pg,
            MODULE_KEYS.ENZYME_ACTIVITY_KEY: enzyme_activity,
        }
        
        
    def loss(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | Distribution | None],
        generative_outputs: dict[str, Distribution | None],
        kl_weight: float = 1.0,
        graph_kl_weight: float = 1.0,
        data_elbo_weight: float = 1.0,
        graph_elbo_weight: float = 1.0,
        rxn_genes_weight: float = 1.0,
        background_to_metabolic_weight: float = 1.0,
    ) -> LossOutput:
        """Compute loss."""
        
        from torch.distributions import kl_divergence

        def old_loss(
            self,
            tensors,
            inference_outputs,
            generative_outputs,
            kl_weight,
            graph_kl_weight,
            data_elbo_weight,
            graph_elbo_weight,
            rxn_genes_weight,
            background_to_metabolic_weight,
        ):
            x = tensors['cells'][REGISTRY_KEYS.X_KEY]
            v = inference_outputs[MODULE_KEYS.V_KEY]
            kl_divergence_m = kl_divergence(
                inference_outputs[MODULE_KEYS.QM_KEY], generative_outputs[MODULE_KEYS.PM_KEY]
            ).sum(dim=-1) / x.shape[1]
            kl_divergence_b = kl_divergence(
                inference_outputs[MODULE_KEYS.QB_KEY], generative_outputs[MODULE_KEYS.PB_KEY]
            ).sum(dim=-1) / x.shape[1]
            kl_divergence_v = kl_divergence(
                inference_outputs[MODULE_KEYS.QV_KEY], generative_outputs[MODULE_KEYS.PV_KEY]
            ).sum(dim=-1).mean() / v.shape[0]
            
            if not self.use_observed_lib_size:
                kl_divergence_lm = kl_divergence(
                    inference_outputs[MODULE_KEYS.QLM_KEY], generative_outputs[MODULE_KEYS.PLM_KEY]
                ).sum(dim=1)
                kl_divergence_lb = kl_divergence(
                    inference_outputs[MODULE_KEYS.QLB_KEY], generative_outputs[MODULE_KEYS.PLB_KEY]
                ).sum(dim=1)
            else:
                kl_divergence_lm = torch.zeros_like(kl_divergence_m)
                kl_divergence_lb = torch.zeros_like(kl_divergence_b)
            
            x_nll_all = -generative_outputs[MODULE_KEYS.PX_KEY].log_prob(x)
            x_nll = x_nll_all.mean(dim=-1)

            kl_local_for_warmup = kl_divergence_m + kl_divergence_b
            kl_local_no_warmup = kl_divergence_lm + kl_divergence_lb
            weighted_kl_local = kl_weight * kl_local_for_warmup + kl_local_no_warmup
            x_elbo = x_nll + weighted_kl_local

            ewt = tensors['graph'][GRAPH_REGISTRY_KEYS.EWT_KEY]
            g_nll = -generative_outputs[MODULE_KEYS.PG_KEY].log_prob(ewt)
            pos_mask = (ewt != 0).to(torch.int64)
            n_pos = pos_mask.sum()
            n_neg = pos_mask.numel() - n_pos
            g_nll_pn = torch.zeros(2, dtype=g_nll.dtype, device=g_nll.device)
            g_nll_pn.scatter_add_(0, pos_mask, g_nll)
            avgc = (n_pos > 0).to(torch.int64) + (n_neg > 0).to(torch.int64)
            g_nll = (g_nll_pn[0] / max(n_neg, 1) + g_nll_pn[1] / max(n_pos, 1)) / avgc

            if self.fixed_graph_cell_kl:
                g_elbo = g_nll + kl_weight * kl_divergence_v
            else:
                g_elbo = g_nll + graph_kl_weight * kl_divergence_v

            px_b = generative_outputs[MODULE_KEYS.PX_BACK_KEY]
            px_b_means = px_b.scale

            if not self.strict_met_back_separation:
                background_to_metabolic_loss = ((px_b_means[:,self.gene_is_metabolic])**2).sum(dim=-1).mean()
                rxns_to_genes_loss = self.decoder.rxns_to_genes_loss()
            else:
                background_to_metabolic_loss = 0
                rxns_to_genes_loss = 0

            metabolic_nll = x_nll_all[:, self.gene_is_metabolic].mean(dim=-1).mean()
            background_nll = x_nll_all[:, ~self.gene_is_metabolic].mean(dim=-1).mean()

            losses = {
                "kl_m_old": kl_divergence_m,
                "kl_b_old": kl_divergence_b,
                "kl_v_old": kl_divergence_v,
                "kl_lm_old": kl_divergence_lm,
                "kl_lb_old": kl_divergence_lb,
                "x_nll_old": x_nll,
                "g_nll_old": g_nll,
                "x_elbo_old": x_elbo,
                "g_elbo_old": g_elbo,
                "background_to_metabolic_loss_old": background_to_metabolic_loss,
                "rxns_to_genes_loss_old": rxns_to_genes_loss,
                "metabolic_nll_old": metabolic_nll,
                "background_nll_old": background_nll,
                "loss_old": data_elbo_weight * x_elbo.mean() \
                    + graph_elbo_weight * g_elbo \
                    + rxn_genes_weight * rxns_to_genes_loss \
                    + background_to_metabolic_weight * background_to_metabolic_loss
            }

            return losses
        
        old_losses = old_loss(self, tensors, inference_outputs, generative_outputs, kl_weight, graph_kl_weight, data_elbo_weight, graph_elbo_weight, rxn_genes_weight, background_to_metabolic_weight)

        x = tensors['cells'][REGISTRY_KEYS.X_KEY]
        v_shape = inference_outputs[MODULE_KEYS.V_KEY].shape
        m_shape = inference_outputs[MODULE_KEYS.M_KEY].shape
        b_shape = inference_outputs[MODULE_KEYS.B_KEY].shape
        kl_divergence_m = kl_divergence(
            inference_outputs[MODULE_KEYS.QM_KEY], generative_outputs[MODULE_KEYS.PM_KEY]
        ).sum(dim=-1) / m_shape[1]
        kl_divergence_b = kl_divergence(
            inference_outputs[MODULE_KEYS.QB_KEY], generative_outputs[MODULE_KEYS.PB_KEY]
        ).sum(dim=-1) / b_shape[1]
        kl_divergence_v = kl_divergence(
            inference_outputs[MODULE_KEYS.QV_KEY], generative_outputs[MODULE_KEYS.PV_KEY]
        ).sum(dim=-1) / v_shape[1]
        
        if not self.use_observed_lib_size:
            kl_divergence_lm = kl_divergence(
                inference_outputs[MODULE_KEYS.QLM_KEY], generative_outputs[MODULE_KEYS.PLM_KEY]
            ).sum(dim=1)
            kl_divergence_lb = kl_divergence(
                inference_outputs[MODULE_KEYS.QLB_KEY], generative_outputs[MODULE_KEYS.PLB_KEY]
            ).sum(dim=1)
        else:
            kl_divergence_lm = torch.zeros_like(kl_divergence_m)
            kl_divergence_lb = torch.zeros_like(kl_divergence_b)
        
        x_nll_all = -generative_outputs[MODULE_KEYS.PX_KEY].log_prob(x)
        x_nll = x_nll_all.sum(dim=-1) / x.shape[1]

        kl_local_for_warmup = kl_divergence_m + kl_divergence_b
        kl_local_no_warmup = kl_divergence_lm + kl_divergence_lb
        weighted_kl_local = kl_weight * kl_local_for_warmup + kl_local_no_warmup
        # maybe add kl weight
        x_elbo = torch.mean(x_nll + weighted_kl_local)

        ewt = tensors['graph'][GRAPH_REGISTRY_KEYS.EWT_KEY]
        g_nll = -generative_outputs[MODULE_KEYS.PG_KEY].log_prob(ewt)
        g_nll = torch.mean(g_nll)
        kl_divergence_v = torch.mean(kl_divergence_v)

        if self.fixed_graph_cell_kl:
            g_elbo = g_nll + kl_weight * kl_divergence_v
        else:
            g_elbo = g_nll + graph_kl_weight * kl_divergence_v

        px_b = generative_outputs[MODULE_KEYS.PX_BACK_KEY]
        px_b_means = px_b.scale

        if not self.strict_met_back_separation:
            background_to_metabolic_loss = ((px_b_means[:,self.gene_is_metabolic])**2).sum(dim=-1).mean()
            rxns_to_genes_loss = self.decoder.rxns_to_genes_loss()
        else:
            background_to_metabolic_loss = 0
            rxns_to_genes_loss = 0

        # calculate metabolic genes loss and background genes loss for tracking
        metabolic_nll = x_nll_all[:, self.gene_is_metabolic].sum(dim=-1).mean()
        background_nll = x_nll_all[:, ~self.gene_is_metabolic].sum(dim=-1).mean()

        loss = data_elbo_weight * x_elbo \
            + graph_elbo_weight * g_elbo \
            + rxn_genes_weight * rxns_to_genes_loss \
            + background_to_metabolic_weight * background_to_metabolic_loss
        loss = torch.mean(loss)
        
        extra_metrics={
                "g_nll": g_nll,
                "kl_v": kl_divergence_v,
                "background_to_metabolic_loss": background_to_metabolic_loss,
                "rxns_to_genes_loss": rxns_to_genes_loss,
                "metabolic_nll": metabolic_nll,
                "background_nll": background_nll,
        }

        extra_metrics.update(old_losses)

        return LossOutput(
            loss=loss,
            reconstruction_loss=x_nll,
            kl_local={
                MODULE_KEYS.KL_M_KEY: kl_divergence_m,
                MODULE_KEYS.KL_B_KEY: kl_divergence_b,
                MODULE_KEYS.KL_LM_KEY: kl_divergence_lm,
                MODULE_KEYS.KL_LB_KEY: kl_divergence_lb,
            },
            extra_metrics=extra_metrics,
        )
        
