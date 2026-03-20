"""MERN model for single-cell data analysis."""

from typing import Optional, Union, Literal
import warnings
import anndata
from scvi._types import AnnOrMuData, Number
import torch
import pandas as pd
import numpy as np
from anndata import AnnData
import networkx as nx
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import Data

from scvi import REGISTRY_KEYS, settings
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
)
from scvi.train import (
    TrainingPlan,
    TrainRunner,
)
from scvi.model.base import BaseModelClass, RNASeqMixin, VAEMixin
from scvi.model._utils import _init_library_size, get_max_epochs_heuristic, use_distributed_sampler, _get_batch_code_from_category
from scvi.utils import setup_anndata_dsp, track
from scvi.utils._docstrings import devices_dsp
from scvi.data._utils import _validate_adata_dataloader_input
from scvi.distributions._utils import DistributionConcatenator, subset_distribution

from .module import MERNModule
from .mern_data_splitting import MERNDataSplitter
from .mern_dataloader import MERNDataLoader
from .custom_optimizer import create_rmsprop_optimizer

from collections.abc import Iterator, Sequence

import numpy.typing as npt
from anndata import AnnData
from torch import Tensor
from torch.distributions import Distribution


class MERN(VAEMixin, RNASeqMixin, BaseModelClass):
    """MERN model for single-cell data analysis.
    
    This model implements the MERN method for single-cell data analysis.

    adata
        AnnData object that has been registered via :meth:`~scvi.model.SCVI.setup_anndata`. If
        ``None``, then the underlying module will not be initialized until training, and a
        :class:`~lightning.pytorch.core.LightningDataModule` must be passed in during training
        (``EXPERIMENTAL``).
    graph
        NetworkX graph or PyG Data object containing the metabolic network.
    rxn_to_genes
        Dictionary mapping reaction IDs to lists of gene IDs.
    n_hidden
        Number of nodes per hidden layer.
    n_layers
        Number of hidden layers used for encoder and decoder NNs.
    n_metabolic_dim
        Dimensionality of the metabolic latent space.
    n_background_dim
        Dimensionality of the background latent space.
    dropout_rate
        Dropout rate for neural networks.
    dispersion
        One of the following:

        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
    gene_likelihood
        One of:

        * ``'nb'`` - Negative binomial distribution
    use_observed_lib_size
        If ``True``, use the observed library size for RNA as the scaling factor in the mean of the
        conditional distribution.
    vertex_mapping
        Dictionary mapping vertex indices to vertex names. Required if graph is a PyG Data object.
    **kwargs
        Additional keyword arguments for :class:`~scvi.module.MERNModule`.
    """
    
    _module_cls = MERNModule
    _data_splitter_cls = MERNDataSplitter
    _data_loader_cls = MERNDataLoader
    _training_plan_cls = TrainingPlan
    _train_runner_cls = TrainRunner
    
    def __init__(
        self,
        adata: AnnData,
        graph: nx.Graph | Data,
        rxn_to_genes: dict[str, list[str]],
        registry: dict | None = None,
        n_hidden: int = 128,
        n_layers: int = 2,
        n_metabolic_dim: int = 10,
        n_background_dim: int = 10,
        dropout_rate: float = 0.1,
        dispersion: Literal["gene"] = "gene",
        gene_likelihood: Literal["nb"] = "nb",
        use_observed_lib_size: bool = True,
        latent_distribution: Literal["normal", "ln"] = "normal",
        vertex_mapping: dict[int, str] | None = None,
        positive_met_dims: bool = False,
        fixed_rxn_genes: bool = False,
        strict_met_back_separation: bool = False,
        rxn_genes_bias: bool = False,
        **kwargs,
    ):
        """Initialize MERN model.
        
        Args:
            adata: AnnData object that has been registered via `setup_anndata`
            n_hidden: Number of hidden units
            n_layers: Number of layers
            dropout_rate: Dropout rate
            **model_kwargs: Additional keyword arguments for the model
        """
        super().__init__(adata, registry)

        met_genes = []
        for rxn in rxn_to_genes:
            temp_genes = set(rxn_to_genes[rxn])
            met_genes+=list(temp_genes)
        met_genes = list(set(met_genes))

        met_label = []
        for gene in adata.var_names:
            if gene in met_genes:
                met_label.append('Metabolic')
            else:
                met_label.append('Non-metabolic')
        
        self.metabolic_genes = pd.Series(met_label, index=adata.var_names)

        if (self.metabolic_genes=='Metabolic').sum() < 200:
            warnings.warn("Less than 200 metabolic genes found. You may not be including relevant genes.")
        
        # handle graph conversion here
        if isinstance(graph, Data) and vertex_mapping is None:
            raise ValueError("Vertex mapping must be provided if graph is a PyG Data object.")
        
        if isinstance(graph, nx.Graph):
            # this is the same logic as from_networkx so should work as long as graph doesn't change between calls
            self.vertex_mapping = dict(zip(range(graph.number_of_nodes()), graph.nodes()))
            nx_graph = from_networkx(graph, group_edge_attrs=["weight", "sign"])
            self.graph = nx_graph
        else:
            self.vertex_mapping = vertex_mapping
            self.graph = graph
        
        self.vertex_names_ordered = [self.vertex_mapping[i] for i in range(len(self.vertex_mapping))]
        self.gene_names_ordered = adata.var_names
        self.rxn_to_genes = rxn_to_genes

        self._model_summary_string = (
            "MERN model with the following parameters:\n"
            f"n_hidden: {n_hidden}, n_layers: {n_layers}, dropout_rate: {dropout_rate}"
            f"n_metabolic_dim: {n_metabolic_dim}, n_background_dim: {n_background_dim}"
            f"dispersion: {dispersion}, gene_likelihood: {gene_likelihood}"
            f"use_observed_lib_size: {use_observed_lib_size}, latent_distribution: {latent_distribution}"
        )
        
        n_cats_per_cov = (
                    self.adata_manager.get_state_registry(
                        REGISTRY_KEYS.CAT_COVS_KEY
                    ).n_cats_per_key
                    if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
                    else None
        )   
        n_batch = self.summary_stats.n_batch
        use_size_factor_key = self.registry_["setup_args"][
            f"{REGISTRY_KEYS.SIZE_FACTOR_KEY}_key"
        ]
        library_log_means, library_log_vars = None, None
        if (
            not use_size_factor_key
            and not use_observed_lib_size
        ):
            raise ValueError("learning library size is not supported for MERN currently")
            library_log_means, library_log_vars = _init_library_size(
                self.adata_manager, n_batch
            )
        self.module = self._module_cls(
            genes=self.gene_names_ordered,
            vertices=self.vertex_mapping,
            rxn_to_genes=rxn_to_genes,
            metabolic_genes=self.metabolic_genes,
            n_input=self.summary_stats.n_vars,
            n_batch=n_batch,
            n_labels=self.summary_stats.n_labels,
            n_continuous_cov=self.summary_stats.get("n_extra_continuous_covs", 0),
            n_hidden=n_hidden,
            n_metabolic_dim=n_metabolic_dim,
            n_background_dim=n_background_dim,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            use_observed_lib_size=use_observed_lib_size,
            latent_distribution=latent_distribution,
            use_size_factor_key=use_size_factor_key,
            positive_met_dims=positive_met_dims,
            fixed_rxn_genes=fixed_rxn_genes,
            strict_met_back_separation=strict_met_back_separation,
            rxn_genes_bias=rxn_genes_bias,
            **kwargs,
        )
        
        self.init_params_ = self._get_init_params(locals())
    
    @devices_dsp.dedent
    def train(
        self,
        max_epochs: int | None = None,
        accelerator: str = "auto",
        devices: int | list[int] | str = "auto",
        train_size: float | None = None,
        validation_size: float | None = None,
        shuffle_set_split: bool = True,
        load_sparse_tensor: bool = False,
        batch_size: int = 128,
        early_stopping: bool = False,
        datasplitter_kwargs: dict | None = None,
        plan_kwargs: dict | None = None,
        **trainer_kwargs,
    ):
        """Train the model.

        Parameters
        ----------
        max_epochs
            The maximum number of epochs to train the model. The actual number of epochs may be
            less if early stopping is enabled. If ``None``, defaults to a heuristic based on
            :func:`~scvi.model.get_max_epochs_heuristic`. Must be passed in if ``datamodule`` is
            passed in, and it does not have an ``n_obs`` attribute.
        %(param_accelerator)s
        %(param_devices)s
        train_size
            Float, or None. Size of training set in the range ``[0.0, 1.0]``. default is None,
            which is practicaly 0.9 and potentially adding small last batch to validation cells.
            Passed into :class:`~scvi.dataloaders.DataSplitter`.
            Not used if ``datamodule`` is passed in.
        validation_size
            Size of the test set. If ``None``, defaults to ``1 - train_size``. If
            ``train_size + validation_size < 1``, the remaining cells belong to a test set. Passed
            into :class:`~scvi.dataloaders.DataSplitter`. Not used if ``datamodule`` is passed in.
        shuffle_set_split
            Whether to shuffle indices before splitting. If ``False``, the val, train, and test set
            are split in the sequential order of the data according to ``validation_size`` and
            ``train_size`` percentages. Passed into :class:`~scvi.dataloaders.DataSplitter`. Not
            used if ``datamodule`` is passed in.
        load_sparse_tensor
            ``EXPERIMENTAL`` If ``True``, loads data with sparse CSR or CSC layout as a
            :class:`~torch.Tensor` with the same layout. Can lead to speedups in data transfers to
            GPUs, depending on the sparsity of the data. Passed into
            :class:`~scvi.dataloaders.DataSplitter`. Not used if ``datamodule`` is passed in.
        batch_size
            Minibatch size to use during training. Passed into
            :class:`~scvi.dataloaders.DataSplitter`. Not used if ``datamodule`` is passed in.
        early_stopping
            Perform early stopping. Additional arguments can be passed in through ``**kwargs``.
            See :class:`~scvi.train.Trainer` for further options.
        datasplitter_kwargs
            Additional keyword arguments passed into :class:`~scvi.dataloaders.DataSplitter`.
            Values in this argument can be overwritten by arguments directly passed into this
            method, when appropriate. Not used if ``datamodule`` is passed in.
        plan_kwargs
            Additional keyword arguments passed into :class:`~scvi.train.TrainingPlan`. Values in
            this argument can be overwritten by arguments directly passed into this method, when
            appropriate.
        **kwargs
           Additional keyword arguments passed into :class:`~scvi.train.Trainer`.
        """
        if max_epochs is None:
            max_epochs = get_max_epochs_heuristic(self.adata.n_obs)
            
        datasplitter_kwargs = datasplitter_kwargs or {}
        datamodule = self._data_splitter_cls(
            self.adata_manager,
            self.graph,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            shuffle_set_split=shuffle_set_split,
            distributed_sampler=use_distributed_sampler(trainer_kwargs.get("strategy", None)),
            load_sparse_tensor=load_sparse_tensor,
            **datasplitter_kwargs,
        )
        
        if self.module is None:
            raise ValueError("Module is not initialized. Please create instance of MERN first.")
        
        plan_kwargs = plan_kwargs or {}
        if "optimizer" not in plan_kwargs:
            plan_kwargs["optimizer"] = "Custom"
            plan_kwargs["optimizer_creator"] = create_rmsprop_optimizer

        training_plan = self._training_plan_cls(self.module, **plan_kwargs)

        es = "early_stopping"
        trainer_kwargs[es] = (
            early_stopping if es not in trainer_kwargs.keys() else trainer_kwargs[es]
        )
        runner = self._train_runner_cls(
            self,
            training_plan=training_plan,
            data_splitter=datamodule,
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            **trainer_kwargs,
        )
        return runner()

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: str | None = None,
        batch_key: str | None = None,
        labels_key: str | None = None,
        size_factor_key: str | None = None,
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        **kwargs,
    ):
        """%(summary)s.

        Parameters
        ----------
        %(param_adata)s
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
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            NumericalObsField(REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False),
            CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys),
        ]
        
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    @torch.inference_mode()
    def get_latent_representation(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        give_mean: bool = True,
        mc_samples: int = 5_000,
        batch_size: int | None = None,
        return_dist: bool = False,
        dataloader: Iterator[dict[str, Tensor | None]] = None,
    ) -> npt.NDArray | tuple[npt.NDArray, npt.NDArray]:
        """Compute the latent representation of the data.

        This is typically denoted as :math:`z_n`.

        Parameters
        ----------
        adata
            :class:`~anndata.AnnData` object with :attr:`~anndata.AnnData.var_names` in the same
            order as the ones used to train the model. If ``None`` and ``dataloader`` is also
            ``None``, it defaults to the object used to initialize the model.
        indices
            Indices of observations in ``adata`` to use. If ``None``, defaults to all observations.
            Ignored if ``dataloader`` is not ``None``
        give_mean
            If ``True``, returns the mean of the latent distribution. If ``False``, returns an
            estimate of the mean using ``mc_samples`` Monte Carlo samples.
        mc_samples
            Number of Monte Carlo samples to use for the estimator for distributions with no
            closed-form mean (e.g., the logistic normal distribution). Not used if ``give_mean`` is
            ``True`` or if ``return_dist`` is ``True``.
        batch_size
            Minibatch size for the forward pass. If ``None``, defaults to
            ``scvi.settings.batch_size``. Ignored if ``dataloader`` is not ``None``
        return_dist
            If ``True``, returns the mean and variance of the latent distribution. Otherwise,
            returns the mean of the latent distribution.
        dataloader
            An iterator over minibatches of data on which to compute the metric. The minibatches
            should be formatted as a dictionary of :class:`~torch.Tensor` with keys as expected by
            the model. If ``None``, a dataloader is created from ``adata``.

        Returns
        -------
        An array of shape ``(n_obs, n_latent)`` if ``return_dist`` is ``False``. Otherwise, returns
        a tuple of arrays ``(n_obs, n_latent)`` with the mean and variance of the latent
        distribution.
        """
        from torch.distributions import Normal
        from torch.nn.functional import softmax

        from .constants import MODULE_KEYS

        self._check_if_trained(warn=False)
        _validate_adata_dataloader_input(self, adata, dataloader)

        if dataloader is None:
            adata = self._validate_anndata(adata)
            dataloader = self._make_data_loader(
                adata=adata, indices=indices, batch_size=batch_size
            )
        else:
            for param in [indices, batch_size]:
                if param is not None:
                    Warning(
                        f"Using {param} after custom Dataloader was initialize is redundant, "
                        f"please re-initialize with selected {param}",
                    )

        ms: list[Tensor] = []
        qm_means: list[Tensor] = []
        qm_vars: list[Tensor] = []
        bs: list[Tensor] = []
        qb_means: list[Tensor] = []
        qb_vars: list[Tensor] = []
        vs: list[Tensor] = []
        qv_means: list[Tensor] = []
        qv_vars: list[Tensor] = []

        for tensors in dataloader:
            outputs: dict[str, Tensor | Distribution | None] = self.module.inference(
                **self.module._get_inference_input(tensors)
            )

            if MODULE_KEYS.QM_KEY in outputs and MODULE_KEYS.QB_KEY and MODULE_KEYS.QV_KEY in outputs:
                qm: Distribution = outputs.get(MODULE_KEYS.QM_KEY)
                qmm: Tensor = qm.loc
                qmv: Tensor = qm.scale.square()
                qb: Distribution = outputs.get(MODULE_KEYS.QB_KEY)
                qbm: Tensor = qb.loc
                qbv: Tensor = qb.scale.square()
                qv: Distribution = outputs.get(MODULE_KEYS.QV_KEY)
                qvm: Tensor = qv.loc
                qvv: Tensor = qv.scale.square() 
            else:
                raise ValueError("Inference distributions missing")

            if return_dist:
                qm_means.append(qmm.cpu())
                qm_vars.append(qmv.cpu())
                qb_means.append(qbm.cpu())
                qb_vars.append(qbv.cpu())
                qv_means.append(qvm.cpu())
                qv_vars.append(qvv.cpu())
                continue

            m: Tensor = qmm if give_mean else outputs.get(MODULE_KEYS.M_KEY)
            b: Tensor = qbm if give_mean else outputs.get(MODULE_KEYS.B_KEY)
            v: Tensor = qvm if give_mean else outputs.get(MODULE_KEYS.V_KEY)

            if give_mean and getattr(self.module, "latent_distribution", None) == "ln":
                samples = m.sample([mc_samples])
                m = softmax(samples, dim=-1).mean(dim=0)
                samples = b.sample([mc_samples])
                b = softmax(samples, dim=-1).mean(dim=0)
                samples = v.sample([mc_samples])
                v = softmax(samples, dim=-1).mean(dim=0)

            ms.append(m.cpu())
            bs.append(b.cpu())
            vs.append(v.cpu())

        # only need to take one entry for graph since same every batch
        if return_dist:
            return torch.cat(qm_means).numpy(), torch.cat(qm_vars).numpy(), torch.cat(qb_means).numpy(), torch.cat(qb_vars).numpy(), qv_means[0].numpy(), qv_vars[0].numpy()
        else:
            return torch.cat(ms).numpy(), torch.cat(bs).numpy(), vs[0].numpy()
        
    @torch.inference_mode()
    def get_normalized_expression(
        self,
        adata: AnnData | None = None,
        indices: list[int] | None = None,
        transform_batch: list[Number | str] | None = None,
        gene_list: list[str] | None = None,
        library_size: float | Literal["latent"] = 1,
        n_samples: int = 1,
        n_samples_overall: int = None,
        weights: Literal["uniform"] | None = None,
        batch_size: int | None = None,
        return_mean: bool = True,
        return_numpy: bool | None = None,
        silent: bool = True,
        dataloader: Iterator[dict[str, Tensor | None]] | None = None,
    ) -> np.ndarray | pd.DataFrame:
        r"""Returns the normalized (decoded) gene expression.

        This is denoted as :math:`\rho_n` in the scVI paper.

        Impotance weights not supported for MERN.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        transform_batch
            Batch to condition on.
            If transform_batch is:
            - None, then real observed batch is used.
            - int, then batch transform_batch is used.
            - Otherwise based on string
        gene_list
            Return frequencies of expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        library_size
            Scale the expression frequencies to a common library size.
            This allows gene expression levels to be interpreted on a common scale of relevant
            magnitude. If set to `"latent"`, use the latent library size.
        n_samples
            Number of posterior samples to use for estimation.
        n_samples_overall
            Number of posterior samples to use for estimation. Overrides `n_samples`.
        weights
            Weights to use for sampling. If `None`, defaults to `"uniform"`.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame
            includes gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults
            to `False`. Otherwise, it defaults to `True`.
        %(de_silent)s
        dataloader
            An iterator over minibatches of data on which to compute the metric. The minibatches
            should be formatted as a dictionary of :class:`~torch.Tensor` with keys as expected by
            the model. If ``None``, a dataloader is created from ``adata``.

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
        _validate_adata_dataloader_input(self, adata, dataloader)

        if dataloader is None:
            adata = self._validate_anndata(adata)

            if indices is None:
                indices = np.arange(adata.n_obs)
            if n_samples_overall is not None:
                assert n_samples == 1  # default value
                n_samples = n_samples_overall // len(indices) + 1
            scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

            transform_batch = _get_batch_code_from_category(
                self.get_anndata_manager(adata, required=True), transform_batch
            )

            gene_mask = slice(None) if gene_list is None else adata.var_names.isin(gene_list)

        else:
            scdl = dataloader
            for param in [indices, batch_size, n_samples]:
                if param is not None:
                    Warning(
                        f"Using {param} after custom Dataloader was initialize is redundant, "
                        f"please re-initialize with selected {param}",
                    )
            gene_mask = slice(None)
            transform_batch = [None]

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "`return_numpy` must be `True` if `n_samples > 1` and `return_mean` "
                    "is`False`, returning an `np.ndarray`.",
                    UserWarning,
                    stacklevel=settings.warnings_stacklevel,
                )
            return_numpy = True
        if library_size == "latent":
            generative_output_key = "mu"
            scaling = 1
        else:
            generative_output_key = "scale"
            scaling = library_size
        
        if generative_output_key == "scale": raise NotImplementedError("Scale not supported for MERN.")

        store_distributions = weights == "importance"
        if store_distributions and len(transform_batch) > 1:
            raise NotImplementedError(
                "Importance weights cannot be computed when expression levels are averaged across "
                "batches."
            )

        exprs = []
        px_store = DistributionConcatenator()
        for tensors in scdl:
            per_batch_exprs = []
            for batch in track(transform_batch, disable=silent):
                generative_kwargs = self._get_transform_batch_gen_kwargs(batch)
                inference_kwargs = {"n_samples": n_samples}
                inference_outputs, generative_outputs = self.module.forward(
                    tensors=tensors,
                    inference_kwargs=inference_kwargs,
                    generative_kwargs=generative_kwargs,
                    compute_loss=False,
                )
                px_generative = generative_outputs["px"]
                if isinstance(px_generative, torch.Tensor):
                    exp_ = px_generative
                else:
                    exp_ = px_generative.get_normalized(generative_output_key)
                exp_ = exp_[..., gene_mask]
                exp_ *= scaling
                per_batch_exprs.append(exp_[None].cpu())
                if store_distributions:
                    px_store.store_distribution(generative_outputs["px"])

            per_batch_exprs = torch.cat(per_batch_exprs, dim=0).mean(0).numpy()
            exprs.append(per_batch_exprs)

        cell_axis = 1 if n_samples > 1 else 0
        exprs = np.concatenate(exprs, axis=cell_axis)

        if n_samples_overall is not None:
            # Converts the 3d tensor to a 2d tensor
            exprs = exprs.reshape(-1, exprs.shape[-1])
            n_samples_ = exprs.shape[0]
            if (weights is None) or weights == "uniform":
                p = None
            else:
                raise NotImplementedError("Importance weights not supported for MERN.")
            ind_ = np.random.choice(n_samples_, n_samples_overall, p=p, replace=True)
            exprs = exprs[ind_]
        elif n_samples > 1 and return_mean:
            exprs = exprs.mean(0)

        if (return_numpy is None or return_numpy is False) and dataloader is None:
            return pd.DataFrame(
                exprs,
                columns=adata.var_names[gene_mask],
                index=adata.obs_names[indices],
            )
        else:
            return exprs
        
    @torch.inference_mode()
    def get_decoding(
        self,
        adata: AnnData | None = None,
        indices: list[int] | None = None,
        return_dists: bool = False,
        transform_batch: list[Number | str] | None = None,
        batch_size: int | None = None,
        silent: bool = True,
        return_numpy: bool = False,
        dataloader: Iterator[dict[str, Tensor | None]] | None = None,
        n_samples: int = 1,
    ):
        """Decodes data getting normalized expression, enzyme activity, ...

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        return_dists
            Whether to return the distributions of metabolic and background contributions
        transform_batch
            Batch to condition on.
            If transform_batch is:
            - None, then real observed batch is used.
            - int, then batch transform_batch is used.
            - Otherwise based on string
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        %(de_silent)s
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame
            includes gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults
            to `False`. Otherwise, it defaults to `True`.
        dataloader
            An iterator over minibatches of data on which to compute the metric. The minibatches
            should be formatted as a dictionary of :class:`~torch.Tensor` with keys as expected by
            the model. If ``None``, a dataloader is created from ``adata``.


        Returns
        -------
        A dictionary with the following keys:
        - "enzyme_activity": enzyme activity
        - "met_mu": metabolic mean
        - "met_scale": metabolic rate contribution
        - "bg_mu": background mean
        - "bg_scale": background rate contribution
        - "met_mu": full mean

        if return_dists:
        - "enzyme_activity": enzyme activity
        - "met_dist": metabolic distribution
        - "bg_dist": background distribution
        - "dist": full distribution
        """

        if n_samples > 1:
            raise NotImplementedError("n_samples > 1 not supported for MERN.")
        
        _validate_adata_dataloader_input(self, adata, dataloader)

        if dataloader is None:
            adata = self._validate_anndata(adata)

            if indices is None:
                indices = np.arange(adata.n_obs)
            
            scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

            transform_batch = _get_batch_code_from_category(
                self.get_anndata_manager(adata, required=True), transform_batch
            )

        else:
            scdl = dataloader
            for param in [indices, batch_size]:
                if param is not None:
                    Warning(
                        f"Using {param} after custom Dataloader was initialize is redundant, "
                        f"please re-initialize with selected {param}",
                    )
            transform_batch = [None]

        met_mu = []
        met_scale = []
        bg_mu = []
        bg_scale = []
        full_mu = []
        enzyme_activity = []
        px_m_store = DistributionConcatenator()
        px_b_store = DistributionConcatenator()
        px_store = DistributionConcatenator()
        for tensors in scdl:
            inference_kwargs = {"n_samples": n_samples}
            inference_outputs, generative_outputs = self.module.forward(
                tensors=tensors,
                inference_kwargs=inference_kwargs,
                compute_loss=False,
            )
           
            px_generative = generative_outputs["px"]
            px_m_generative = generative_outputs["px_met"]
            px_b_generative = generative_outputs["px_back"]
            enzyme_activity_generative = generative_outputs["enzyme_activity"]

            met_mu.append(px_m_generative.get_normalized("mu").cpu())
            met_scale.append(px_m_generative.get_normalized("scale").cpu())
            bg_mu.append(px_b_generative.get_normalized("mu").cpu())
            bg_scale.append(px_b_generative.get_normalized("scale").cpu())
            full_mu.append(px_generative.get_normalized("mu").cpu())
            enzyme_activity.append(enzyme_activity_generative.cpu())

            if return_dists:
                px_m_store.store_distribution(px_m_generative)
                px_b_store.store_distribution(px_b_generative)
                px_store.store_distribution(px_generative)

        cell_axis = 1 if n_samples > 1 else 0

        met_mu = np.concatenate(met_mu, axis=cell_axis)
        met_scale = np.concatenate(met_scale, axis=cell_axis)
        bg_mu = np.concatenate(bg_mu, axis=cell_axis)
        bg_scale = np.concatenate(bg_scale, axis=cell_axis)
        enzyme_activity = np.concatenate(enzyme_activity, axis=cell_axis)
        full_mu = np.concatenate(full_mu, axis=cell_axis)
        
        if not return_numpy:
            met_mu = pd.DataFrame(met_mu, columns=adata.var_names, index=adata.obs_names[indices])
            met_scale = pd.DataFrame(met_scale, columns=adata.var_names, index=adata.obs_names[indices])
            bg_mu = pd.DataFrame(bg_mu, columns=adata.var_names, index=adata.obs_names[indices])
            bg_scale = pd.DataFrame(bg_scale, columns=adata.var_names, index=adata.obs_names[indices])
            full_mu = pd.DataFrame(full_mu, columns=adata.var_names, index=adata.obs_names[indices])
            enzyme_activity = pd.DataFrame(enzyme_activity, columns=self.vertex_names_ordered, index=adata.obs_names[indices])

        if return_dists:
            met_dist = px_m_store.get_concatenated_distributions()
            bg_dist = px_b_store.get_concatenated_distributions()
            full_dist = px_store.get_concatenated_distributions()

            return {
                "enzyme_activity": enzyme_activity,
                "met_mu": met_mu,
                "met_scale": met_scale,
                "bg_mu": bg_mu,
                "bg_scale": bg_scale,
                "full_mu": full_mu,
                "met_dist": met_dist,
                "bg_dist": bg_dist,
                "dist": full_dist,
            }
        else:
            return {
                "enzyme_activity": enzyme_activity,
                "met_mu": met_mu,
                "met_scale": met_scale,
                "bg_mu": bg_mu,
                "bg_scale": bg_scale,
                "full_mu": full_mu,
            }
        
    def get_rxn_genes_weights(
        self, 
        return_numpy: bool = False,
    ):
        """Returns the weights of the rxn to genes layer"""
        weights = self.module.decoder.rxn_gene_layer.linear.weight.detach().clone().cpu().numpy()
        if not return_numpy:
            return pd.DataFrame(weights, columns=self.vertex_names_ordered, index=self.gene_names_ordered)
        else:
            return weights

    def _make_data_loader(
        self,
        adata: AnnOrMuData,
        neg_sampling_ratio: int = 1,
        indices: Sequence[int] | None = None,
        batch_size: int | None = None,
        shuffle: bool = False,
        data_loader_class=None,
        **data_loader_kwargs,
    ):
        """Create a MERNDataLoader object for data iteration.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        shuffle
            Whether observations are shuffled each iteration though
        data_loader_class
            Class to use for data loader
        data_loader_kwargs
            Kwargs to the class-specific data loader class
        """
        adata_manager = self.get_anndata_manager(adata)
        if adata_manager is None:
            raise AssertionError(
                "AnnDataManager not found. Call `self._validate_anndata` prior to calling this "
                "function."
            )

        adata = adata_manager.adata

        if batch_size is None:
            batch_size = settings.batch_size
        if indices is None:
            indices = np.arange(adata.n_obs)
        if data_loader_class is None:
            data_loader_class = self._data_loader_cls

        if "num_workers" not in data_loader_kwargs:
            data_loader_kwargs.update({"num_workers": settings.dl_num_workers})
        if "persistent_workers" not in data_loader_kwargs:
            data_loader_kwargs.update({"persistent_workers": settings.dl_persistent_workers})

        dl = data_loader_class(
            adata_manager,
            self.graph,
            neg_sampling_ratio=neg_sampling_ratio,
            shuffle=shuffle,
            indices=indices,
            batch_size=batch_size,
            **data_loader_kwargs,
        )
        return dl
