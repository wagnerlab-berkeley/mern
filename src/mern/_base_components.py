"""Base components for MERN model."""

from typing import Callable, Iterable, Mapping, Optional

import pandas as pd
import torch
import numpy as np
from torch import nn
from torch.distributions import Normal

from torch_geometric.nn import GCNConv

import networkx as nx
from networkx.algorithms import bipartite

from scvi.nn import FCLayers

def _identity(x):
    return x

class RxnsToGenesLayer(nn.Module):
    """Layer that maps reactions to genes.

    Parameters
    ----------
    genes
        genes in same order as data
    rxns
        dictionary mapping indices to reactions
    rxns_to_genes
        mapping from reactions to genes
    strict_met_back_separation
        whether to separate metabolic and background genes
    bias
        whether to include a bias term
    """
    def __init__(
            self,
            genes: Iterable[str],
            rxns: dict[int, str],
            rxns_to_genes: Mapping[str, str],
            n_cat_list: Iterable[int] = None,
            strict_met_back_separation: bool = False,
            rxn_genes_bias: bool = False,
    ):
        super().__init__()
        self.genes = np.array(genes)
        self.rxns = rxns
        self.rxns_to_genes = rxns_to_genes
        self.strict_met_back_separation = strict_met_back_separation
        n_cat_list = [] if n_cat_list is None else n_cat_list
        self.n_cov = sum(n_cat_list)

        # create mask for weights layer that can't be used for loss
        loss_mask = torch.ones(len(self.genes), len(self.rxns))
        for index, temp_rxn in self.rxns.items():
            temp_genes = self.rxns_to_genes[temp_rxn]
            for gene in temp_genes:
                gene_ind = np.argwhere(self.genes == gene)
                loss_mask[gene_ind, index] = 0
        if self.n_cov > 0:
            loss_mask = torch.cat([loss_mask, torch.zeros(len(self.genes), self.n_cov)], axis=1)
        if self.strict_met_back_separation:
             # create mask for weights layer that can be used 
            weight_mask = torch.zeros(len(self.genes), len(self.rxns))
            for index, temp_rxn in self.rxns.items():
                temp_genes = self.rxns_to_genes[temp_rxn]
                for gene in temp_genes:
                    gene_ind = np.argwhere(self.genes == gene)
                    weight_mask[gene_ind, index] = 1
            if self.n_cov > 0:
                weight_mask = torch.cat([weight_mask, torch.ones(len(self.genes), self.n_cov)], axis=1)
            self.register_buffer("weight_mask", weight_mask)

        # add on bias params to mask, in the case of loss mask it should be 0s (so they don't contribute to loss) and for weight mask it should be 1s (so they can contribute)
        self.register_buffer("loss_mask", loss_mask)
        self.linear = nn.Linear(len(self.rxns) + self.n_cov, len(self.genes), bias=rxn_genes_bias)
        
    def forward(self, x):
        for p in self.linear.parameters():
            p.data.clamp_(0)
        
        if self.strict_met_back_separation:
            self.linear.weight.data = self.linear.weight.data * self.weight_mask
        return self.linear(x)

    def reg_subset_loss(self, norm=1):
        """return l1 loss for weights between genes not previously related to a rxn"""
        if norm==1:
            mod_weights = torch.abs(self.linear.weight)
        elif norm==2:
            mod_weights = self.linear.weight**2
        else: raise Exception(f'Norm={norm} not recognized')
            
        loss = mod_weights * self.loss_mask
        return loss.sum()
    
class RxnsToGenesFixed(nn.Module):
    """Layer that maps reactions to genes.

    Parameters
    ----------
    genes
        genes in same order as data
    rxns
        dictionary mapping indices to reactions
    rxns_to_genes
        mapping from reactions to genes
    """
    def __init__(
            self,
            genes: Iterable[str],
            rxns: dict[int, str],
            rxns_to_genes: Mapping[str, str],
    ):
        super().__init__()
        self.genes = np.array(genes)
        self.rxns = rxns
        self.rxns_to_genes = rxns_to_genes

        # create mask for weights layer that can be used 
        mask = torch.zeros(len(self.genes), len(self.rxns))
        for index, temp_rxn in self.rxns.items():
            temp_genes = self.rxns_to_genes[temp_rxn]
            for gene in temp_genes:
                gene_ind = np.argwhere(self.genes == gene)
                mask[gene_ind, index] = 1
        fixed_weights = torch.ones(len(self.genes), len(self.rxns))
        fixed_weights = fixed_weights * mask
        self.register_buffer("fixed_weights", fixed_weights)
        
    def forward(self, x):
        return torch.nn.functional.linear(x, self.fixed_weights)

class GraphEncoder(nn.Module):
    """Graph encoder based on GCN.

    Parameters
    ----------
    vnum
        number of vertices
    n_output
        number of output dimensions
    """
    def __init__(
            self,
            vnum: int,
            n_output: int,
            return_dist: bool = False,
            positive_met_dims: bool = False,
    ):
        super().__init__()
        self.vrepr = nn.Parameter(torch.randn(vnum, n_output))
        self.conv = GCNConv(n_output, n_output, add_self_loops=True, bias=True)
        self.loc = nn.Linear(n_output, n_output)
        self.var = nn.Linear(n_output, n_output)
        self.return_dist = return_dist
        self.z_transformation = _identity
        self.positive_met_dims = positive_met_dims

    def forward(
            self, 
            eidx: torch.Tensor,
            ewt: torch.Tensor | None,
            esgn: torch.Tensor | None,
    ) -> torch.Tensor:
        """Forward pass."""
        ptr = self.conv(self.vrepr, eidx, ewt)
        ptr = nn.functional.relu(ptr)
        loc = self.loc(ptr)
        if self.positive_met_dims:
            loc = torch.nn.functional.softplus(loc)
        #maybe need eps here
        var = nn.functional.softplus(self.var(ptr)) + 1e-6

        dist = Normal(loc, var.sqrt())
        latent = dist.rsample()
        if self.return_dist:
            return dist, latent
        return loc, var, latent

class GraphDecoder(nn.Module):
    """Simple graph decoder.
    """
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        v: torch.Tensor,
        eidx: torch.Tensor,
        esgn: torch.Tensor,
    ):
        """Forward pass.

        Parameters
        ----------
        v
            tensor of shape (nodes,met_dim)
        eidx
            tensor of shape (nodes,2)
        esgn

        Returns
        --------
        tensor of shape (edges) to parameterize bernoulli distribution
        """
        sidx, tidx = eidx  # Source index and target index
        logits = esgn * (v[sidx] * v[tidx]).sum(dim=1) # should be (edges,1)
        return logits

class DecoderMERN(nn.Module):
    """Decoder for MERN.

    Parameters
    ----------
    genes
        genes in same order as data
    rxns
        dictionary mapping indices to reactions
    background_n_input
        number of input dimensions for background
    rxns_to_genes
        mapping from reactions to genes
    metabolic_genes
        indicate which genes are metabolic genes
    n_cat_list
        List of category membership(s) for this sample
    """

    def __init__(
        self,
        genes: Iterable[str],
        rxns: dict[int, str],
        background_n_input: int,
        rxns_to_genes: Mapping[str, str],
        metabolic_genes: pd.Series,
        n_cat_list: Iterable[int] = None,
        fixed_rxn_genes: bool = False,
        strict_met_back_separation: bool = False,
        rxn_genes_bias: bool = False,
    ):
        super().__init__()
        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []
        self.n_cov = sum(self.n_cat_list)

        self.fixed_rxn_genes = fixed_rxn_genes
        self.strict_met_back_separation = strict_met_back_separation
        if fixed_rxn_genes:
            self.rxn_gene_layer = RxnsToGenesFixed(genes, rxns,rxns_to_genes)
        else:
            self.rxn_gene_layer = RxnsToGenesLayer(genes, rxns,rxns_to_genes, n_cat_list=self.n_cat_list, strict_met_back_separation=strict_met_back_separation, rxn_genes_bias=rxn_genes_bias)
        self.back_linear = nn.Linear(background_n_input + self.n_cov, len(genes))

        # save gene order
        self.gene_is_metabolic = []
        for gene in genes:
            self.gene_is_metabolic.append(True if metabolic_genes[gene] == "Metabolic" else False)
        
        self.gene_is_metabolic = np.array(self.gene_is_metabolic)
       
    def forward(
        self,
        dispersion: str,
        m: torch.Tensor,
        b: torch.Tensor,
        v: torch.Tensor,
        metabolic_library: torch.Tensor,
        background_library: torch.Tensor,
        *cat_list: int,
    ):
        """Forward pass.

        Parameters
        ----------
        m
            tensor of shape (met_dim,)
        b
            tensor of shape (background_n_input,)
        v
            tensor of shape (met_dim,)
        metabolic_library
            metabolic library size
        background_library
            background library size
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        2-tuple of 3-tuple of :py:class:`torch.Tensor`
            parameters for the NB distribution of expression
            (metabolic, background, enzyme activity)
        """
        one_hot_cat_list = []

        if len(self.n_cat_list) > len(cat_list):
            raise ValueError("nb. categorical args provided doesn't match init. params.") 
        
        for n_cat, cat in zip(self.n_cat_list, cat_list, strict=False):
            if n_cat and cat is None:
                raise ValueError("cat not provided while n_cat != 0 in init. params.")
            if n_cat > 1:  # n_cat = 1 will be ignored - no additional information
                if cat.size(1) != n_cat:
                    one_hot_cat = nn.functional.one_hot(cat.squeeze(-1), n_cat)
                else:
                    one_hot_cat = cat  # cat has already been one_hot encoded
                one_hot_cat_list += [one_hot_cat]
        cov_list = one_hot_cat_list

        # decode metabolic to cells x rxns first
        r = m @ v.t()
        # restrict rxn activations to be positive for interpretability
        r = r.clamp(min=0)

        r_batch = torch.cat((r, *cov_list), dim=-1)
        metx = self.rxn_gene_layer(r_batch)

        if self.strict_met_back_separation:
            # use normalization for positive values
            metx[:,~self.gene_is_metabolic] = torch.tensor(0, dtype=metx.dtype)
            metx_scale = metx / (metx.sum(dim=1, keepdim=True) + 1e-6)
        else:
            metx_scale = nn.functional.softmax(metx, dim=1)

        b = torch.cat((b, *cov_list), dim=-1)
        backx = self.back_linear(b)
        if self.strict_met_back_separation:
            backx[:,self.gene_is_metabolic] = torch.tensor(-1e9, dtype=backx.dtype)
        backx_scale = nn.functional.softmax(backx, dim=1)

        metx_rate = torch.exp(metabolic_library) * metx_scale
        backx_rate = torch.exp(background_library) * backx_scale

        # can update this for gene-cell dispersion
        metx_r = None
        backx_r = None

        return ((metx_scale, metx_r, metx_rate), (backx_scale, backx_r, backx_rate), r)
    
    def rxns_to_genes_loss(self, norm=1):
        """return l1 loss for weights between genes not previously related to a rxn"""
        if self.fixed_rxn_genes:
            return 0
        else:
            return self.rxn_gene_layer.reg_subset_loss(norm)
    
# Encoder
class MetabolicEncoder(nn.Module):
    """Encode data of ``n_input`` dimensions into a latent space of ``n_output`` dimensions.

    Uses a fully-connected neural network of ``n_hidden`` layers.

    Added functionality to ensure positive latent space.

    Parameters
    ----------
    n_input
        The dimensionality of the input (data space)
    n_output
        The dimensionality of the output (latent space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    distribution
        Distribution of z
    var_eps
        Minimum value for the variance;
        used for numerical stability
    var_activation
        Callable used to ensure positivity of the variance.
        Defaults to :meth:`torch.exp`.
    return_dist
        Return directly the distribution of z instead of its parameters.
    **kwargs
        Keyword args for :class:`~scvi.nn.FCLayers`
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        distribution: str = "normal",
        var_eps: float = 1e-4,
        var_activation: Callable | None = None,
        return_dist: bool = False,
        positive_met_dims: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.distribution = distribution
        self.var_eps = var_eps
        self.encoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            **kwargs,
        )
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)
        self.return_dist = return_dist
        self.positive_met_dims = positive_met_dims
        if distribution == "ln":
            self.z_transformation = nn.Softmax(dim=-1)
        else:
            self.z_transformation = _identity
        self.var_activation = torch.exp if var_activation is None else var_activation

    def forward(self, x: torch.Tensor, *cat_list: int):
        r"""The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\)
         #. Samples a new value from an i.i.d. multivariate normal
            \\( \\sim Ne(q_m, \\mathbf{I}q_v) \\)

        Parameters
        ----------
        x
            tensor with shape (n_input,)
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        3-tuple of :py:class:`torch.Tensor`
            tensors of shape ``(n_latent,)`` for mean and var, and sample

        """
        # Parameters for latent distribution
        q = self.encoder(x, *cat_list)
        q_m = self.mean_encoder(q)
        if self.positive_met_dims:
            q_m = torch.nn.functional.softplus(q_m)
        q_v = self.var_activation(self.var_encoder(q)) + self.var_eps
        dist = Normal(q_m, q_v.sqrt())
        latent = self.z_transformation(dist.rsample())
        if self.return_dist:
            return dist, latent
        return q_m, q_v, latent
