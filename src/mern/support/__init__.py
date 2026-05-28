"""
MeRN Support Tools

A collection of supporting tools for MeRN development and operations.
"""

from importlib import import_module

from .._version import __version__

__author__ = "Daniel Lewinsohn"
__email__ = "daniel_lewinsohn@berkeley.edu"

# Import functions from existing modules
from ._analysis import (
    latent_auc,
    latent_regression,
    get_gene_sets,
    cohens_d,
    wilcoxon_test,
    random_walk,
    active_custom_pathways,
    pairwise_distance_correlation,
    knn_consistency,
    reaction_correlation,
    calculate_directions,
    calculate_pathway_scores,
    calculate_ddp_scores,
    calculate_ddps,
)

from ._ddp_llm_context import (
    SCHEMA_VERSION as DDP_LLM_SCHEMA_VERSION,
    DEFAULT_SCORE_INTERPRETATION,
    DEFAULT_DDP_INTERPRETATION,
    build_context_guide,
    build_ddp_static_metadata,
    ddp_ids_matching_pathways,
    ddp_ids_matching_reactions,
    export_ddp_llm_bundle,
    export_ddp_context_simple,
    filter_ddp_activity,
    filter_ddp_definitions,
    genes_for_decode_keys,
    build_node_to_ddps,
    neighbor_ddps_for_graph,
    merge_ddp_llm_bundle,
    slim_provenance_for_bundle_export,
    summarize_ddp_obs_context,
    write_ddp_llm_bundle_json,
    write_ddp_llm_bundle_jsonl,
)

from ._metabolic_datasets import *
# from ._addEdge import *

# Import configuration
from .config import KEGG_DIR

_PLOT_EXPORTS = {
    "training_plot",
    "metabolic_topology_plot",
    "metabolic_topology_plot_go",
    "kegg_pathway_plot",
    "custom_pathway_plot",
    "rxn_volcano_plot",
    "reaction_scatter_plot",
}


def __getattr__(name):
    if name in _PLOT_EXPORTS:
        _plots = import_module("._plots", __name__)
        value = getattr(_plots, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _PLOT_EXPORTS)

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    # Analysis functions
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
    "calculate_ddp_scores",
    "calculate_ddps",
    # DDP LLM context export
    "DDP_LLM_SCHEMA_VERSION",
    "DEFAULT_SCORE_INTERPRETATION",
    "DEFAULT_DDP_INTERPRETATION",
    "build_context_guide",
    "build_ddp_static_metadata",
    "ddp_ids_matching_pathways",
    "ddp_ids_matching_reactions",
    "export_ddp_llm_bundle",
    "export_ddp_context_simple",
    "filter_ddp_activity",
    "filter_ddp_definitions",
    "genes_for_decode_keys",
    "build_node_to_ddps",
    "neighbor_ddps_for_graph",
    "merge_ddp_llm_bundle",
    "slim_provenance_for_bundle_export",
    "summarize_ddp_obs_context",
    "write_ddp_llm_bundle_json",
    "write_ddp_llm_bundle_jsonl",
    # Plotting functions
    "training_plot",
    "metabolic_topology_plot",
    "metabolic_topology_plot_go",
    "kegg_pathway_plot",
    "custom_pathway_plot",
    "rxn_volcano_plot",
    "reaction_scatter_plot",
    # Metabolic datasets
    "KeggKGMLMetabolicDataset",
    # Configuration
    "KEGG_DIR",
] 
