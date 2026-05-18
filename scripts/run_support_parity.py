"""Compare old and new mern-support outputs without importing old __init__."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OLD_SUPPORT = ROOT.parent / "mern-support" / "mern_support"
ADATA_PATH = ROOT / "tests" / "data" / "mouse_intestine_100.h5ad"
OUTPUT_PATH = ROOT / "tests" / "repro_outputs" / "support_parity.json"


def _load_legacy_module(package_name: str, module_name: str):
    package = types.ModuleType(package_name)
    package.__path__ = [str(OLD_SUPPORT)]
    sys.modules[package_name] = package

    for name in ("config", "_addEdge"):
        path = OLD_SUPPORT / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"{package_name}.{name}", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"{package_name}.{name}"] = module
        spec.loader.exec_module(module)
        setattr(package, name, module)

    path = OLD_SUPPORT / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(f"{package_name}.{module_name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"{package_name}.{module_name}"] = module
    spec.loader.exec_module(module)
    return module


def _numeric_summary(value: Any) -> dict[str, Any]:
    arr = np.asarray(value, dtype=float)
    return {
        "shape": list(arr.shape),
        "sum": float(np.nansum(arr)),
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr)),
        "nan_count": int(np.isnan(arr).sum()),
    }


def _compare_summary(name: str, old_value: Any, new_value: Any) -> dict[str, Any]:
    old_arr = np.asarray(old_value)
    new_arr = np.asarray(new_value)
    result = {
        "name": name,
        "old": _numeric_summary(old_arr),
        "new": _numeric_summary(new_arr),
        "shape_equal": list(old_arr.shape) == list(new_arr.shape),
    }
    if old_arr.shape == new_arr.shape:
        result["max_abs_diff"] = float(np.nanmax(np.abs(old_arr.astype(float) - new_arr.astype(float))))
    return result


def main() -> None:
    old_analysis = _load_legacy_module("legacy_msupport", "_analysis")
    old_datasets = _load_legacy_module("legacy_msupport", "_metabolic_datasets")

    from mern.support import _analysis as new_analysis
    from mern.support import _metabolic_datasets as new_datasets

    adata_old = ad.read_h5ad(ADATA_PATH)
    adata_new = ad.read_h5ad(ADATA_PATH)

    old_dataset = old_datasets.KeggKGMLMetabolicDataset(species="mouse", capitalize=False)
    new_dataset = new_datasets.KeggKGMLMetabolicDataset(species="mouse", capitalize=False)

    old_dataset.add_module_info(adata_old)
    new_dataset.add_module_info(adata_new)
    old_graph = old_dataset.metabolic_topology(adata_old, self_loops=False)
    new_graph = new_dataset.metabolic_topology(adata_new, self_loops=False)
    old_dataset.add_rxn_module_info(adata_old, old_graph)
    new_dataset.add_rxn_module_info(adata_new, new_graph)

    rng = np.random.default_rng(0)
    embedding1 = rng.normal(size=(adata_old.n_obs, 4))
    embedding2 = embedding1 + rng.normal(scale=0.01, size=embedding1.shape)
    embedding1_df = pd.DataFrame(embedding1, index=adata_old.obs_names)
    embedding2_df = pd.DataFrame(embedding2, index=adata_old.obs_names)
    adata_old.obsm["X_test_latent"] = embedding1
    adata_new.obsm["X_test_latent"] = embedding1
    positive_label = str(adata_old.obs["cell_type"].astype(str).iloc[0])
    reaction_keys = list(adata_old.uns["Reaction Info"].index)
    enzyme_acts = pd.DataFrame(
        rng.normal(size=(adata_old.n_obs, len(reaction_keys))),
        index=adata_old.obs_names,
        columns=reaction_keys,
    )

    old_pathway_scores, _, _ = old_analysis.calculate_pathway_scores(
        enzyme_acts, adata_old.uns["Reaction Info"], min_rxns=1
    )
    new_pathway_scores, _, _ = new_analysis.calculate_pathway_scores(
        enzyme_acts, adata_new.uns["Reaction Info"], min_rxns=1
    )

    group_a = list(adata_old.obs_names[:20])
    group_b = list(adata_old.obs_names[20:40])

    checks = [
        {
            "name": "gene_reactions_equal",
            "passed": adata_old.var["Gene Reactions"].equals(adata_new.var["Gene Reactions"]),
        },
        {
            "name": "metabolic_gene_equal",
            "passed": adata_old.var["Metabolic Gene"].equals(adata_new.var["Metabolic Gene"]),
        },
        {
            "name": "graph_nodes_equal",
            "passed": set(old_graph.nodes()) == set(new_graph.nodes()),
            "old": old_graph.number_of_nodes(),
            "new": new_graph.number_of_nodes(),
        },
        {
            "name": "graph_edges_equal",
            "passed": set(old_graph.edges()) == set(new_graph.edges()),
            "old": old_graph.number_of_edges(),
            "new": new_graph.number_of_edges(),
        },
        _compare_summary(
            "latent_auc",
            old_analysis.latent_auc(adata_old, "cell_type", positive_label, "X_test_latent"),
            new_analysis.latent_auc(adata_new, "cell_type", positive_label, "X_test_latent"),
        ),
        _compare_summary(
            "latent_regression",
            old_analysis.latent_regression(adata_old, "cell_type", positive_label, "X_test_latent"),
            new_analysis.latent_regression(adata_new, "cell_type", positive_label, "X_test_latent"),
        ),
        _compare_summary(
            "pairwise_distance_correlation",
            [old_analysis.pairwise_distance_correlation(embedding1, embedding2, size=1000)],
            [new_analysis.pairwise_distance_correlation(embedding1, embedding2, size=1000)],
        ),
        _compare_summary(
            "knn_consistency",
            [old_analysis.knn_consistency(embedding1_df, embedding2_df, k=5, return_mean=True)],
            [new_analysis.knn_consistency(embedding1_df, embedding2_df, k=5, return_mean=True)],
        ),
        _compare_summary(
            "reaction_correlation",
            old_analysis.reaction_correlation(enzyme_acts.T, (enzyme_acts + 0.01).T),
            new_analysis.reaction_correlation(enzyme_acts.T, (enzyme_acts + 0.01).T),
        ),
        _compare_summary("pathway_scores", old_pathway_scores, new_pathway_scores),
        _compare_summary(
            "wilcoxon_pathway",
            old_analysis.wilcoxon_test(old_pathway_scores.T, group_a, group_b)[
                ["wilcox_stat", "wilcox_pval", "cohens_d", "adjusted_pval"]
            ],
            new_analysis.wilcoxon_test(new_pathway_scores.T, group_a, group_b)[
                ["wilcox_stat", "wilcox_pval", "cohens_d", "adjusted_pval"]
            ],
        ),
    ]

    failures = []
    for check in checks:
        if "passed" in check:
            if not check["passed"]:
                failures.append(check["name"])
        elif check.get("shape_equal") is not True or check.get("max_abs_diff", 0.0) > 1e-8:
            failures.append(check["name"])

    output = {"checks": checks, "failures": failures}
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(f"Wrote {OUTPUT_PATH}")
    if failures:
        print("Support parity failures:", failures)
        raise SystemExit(1)
    print("Support parity passed")


if __name__ == "__main__":
    main()
