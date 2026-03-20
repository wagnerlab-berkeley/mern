"""Standalone MERN package."""

from .analysis import (
    active_custom_pathways,
    calculate_directions,
    calculate_pathway_scores,
    cohens_d,
    get_gene_sets,
    knn_consistency,
    latent_auc,
    latent_regression,
    pairwise_distance_correlation,
    random_walk,
    reaction_correlation,
    wilcoxon_test,
)
from .config import ANNDATA_KEY, KEGG_DIR
from .graph_dataloader import GraphDataLoader
from .mern_dataloader import MERNDataLoader
from .model import MERN
from .module import MERNModule

__version__ = "0.1.0"

try:
    from .datasets import KeggKGMLMetabolicDataset, MetabolicDataset
except ImportError:  # Optional KEGG dependencies are installed via the `kegg` extra.
    KeggKGMLMetabolicDataset = None
    MetabolicDataset = None

__all__ = [
    "__version__",
    "ANNDATA_KEY",
    "KEGG_DIR",
    "MERN",
    "MERNModule",
    "MERNDataLoader",
    "GraphDataLoader",
    "MetabolicDataset",
    "KeggKGMLMetabolicDataset",
    "latent_auc",
    "latent_regression",
    "get_gene_sets",
    "cohens_d",
    "wilcoxon_test",
    "random_walk",
    "active_custom_pathways",
    "pairwise_distance_correlation",
    "knn_consistency",
    "reaction_correlation",
    "calculate_directions",
    "calculate_pathway_scores",
]
