"""Evaluation metric commands for MERN model replicates."""

from __future__ import annotations

import argparse
import itertools
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm


def add_evaluation_parsers(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "evaluate-models",
        aliases=["model-evaluation", "evaluate_models"],
        help="Evaluate consistency metrics across MERN replicate groups",
        description="Evaluate consistency metrics across MERN replicate groups.",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        required=True,
        help="Directory containing condition directories with model replicate subdirectories",
    )
    parser.add_argument(
        "--adata_path",
        "--adata-path",
        type=Path,
        required=True,
        help="Path to AnnData file",
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        default=100,
        help="Number of neighbors for latent KNN consistency",
    )
    parser.add_argument(
        "--graph_neighbors",
        "--graph-neighbors",
        type=int,
        default=25,
        help="Number of neighbors for graph KNN consistency",
    )
    parser.add_argument(
        "--n_cells",
        "--n-cells",
        type=int,
        default=None,
        help="Number of cells to use for metric calculation. Uses all cells if omitted.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=8,
        help="Base random seed for cell subsetting and independent decoding samples",
    )
    parser.add_argument(
        "--species",
        type=str,
        default="mouse",
        help="Species for KeggKGMLMetabolicDataset",
    )
    parser.add_argument(
        "--n_samples",
        "--n-samples",
        type=int,
        default=8,
        help="Number of independent decoding samples to average per model replicate",
    )
    parser.add_argument(
        "--accelerator",
        default="cpu",
        help="Accelerator to use when loading models",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Disable progress bars and per-decode progress messages",
    )
    parser.set_defaults(func=run_evaluate_models_command)

    cross_parser = subparsers.add_parser(
        "evaluate-cross-models",
        aliases=["model-cross-evaluation", "evaluate_cross_models"],
        help="Evaluate consistency metrics between two MERN replicate directories",
        description="Evaluate consistency metrics between two MERN replicate directories.",
    )
    cross_parser.add_argument(
        "--dir1",
        type=Path,
        required=True,
        help="Directory containing first model replicates",
    )
    cross_parser.add_argument(
        "--dir2",
        type=Path,
        required=True,
        help="Directory containing second model replicates",
    )
    cross_parser.add_argument(
        "--adata_path",
        "--adata-path",
        type=Path,
        required=True,
        help="Path to AnnData file for dir1 models",
    )
    cross_parser.add_argument(
        "--adata_path2",
        "--adata-path2",
        type=Path,
        default=None,
        help="Path to AnnData file for dir2 models. Uses --adata_path if omitted.",
    )
    cross_parser.add_argument(
        "--neighbors",
        type=int,
        default=100,
        help="Number of neighbors for latent KNN consistency",
    )
    cross_parser.add_argument(
        "--graph_neighbors",
        "--graph-neighbors",
        type=int,
        default=25,
        help="Number of neighbors for graph KNN consistency",
    )
    cross_parser.add_argument(
        "--n_cells",
        "--n-cells",
        type=int,
        default=None,
        help="Number of cells to use for metric calculation. Uses all cells if omitted.",
    )
    cross_parser.add_argument(
        "--seed",
        type=int,
        default=8,
        help="Random seed for cell subsetting",
    )
    cross_parser.add_argument(
        "--species",
        type=str,
        default="mouse",
        help="Species for KeggKGMLMetabolicDataset",
    )
    cross_parser.add_argument(
        "--accelerator",
        default="cpu",
        help="Accelerator to use when loading models",
    )
    cross_parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Disable progress bars",
    )
    cross_parser.set_defaults(func=run_evaluate_cross_models_command)


def _prepare_adata(adata_path: Path, species: str):
    import anndata as ad

    from mern import support as mern_support

    rna = ad.read_h5ad(adata_path)
    dataset = mern_support.KeggKGMLMetabolicDataset(species=species, capitalize=False)
    dataset.add_module_info(rna)

    features_to_use = list(
        rna.var_names[rna.var["highly_variable_metabolic"] | rna.var["highly_variable_background"]]
    )
    features_to_use.sort()
    rna = rna[:, features_to_use].copy()

    met_g = dataset.metabolic_topology(rna, self_loops=False)
    dataset.add_rxn_module_info(rna, met_g)
    dataset.get_rxn_genes_all()
    return rna


def _condition_dirs(models_dir: Path) -> list[Path]:
    return sorted(path for path in models_dir.iterdir() if path.is_dir())


def _load_mern_model(model_dir: Path, adata, accelerator: str):
    import mern

    return mern.MERN.load(model_dir, adata, accelerator=accelerator)


def _read_embeddings(model_dir: Path) -> dict[str, Any]:
    with (model_dir / "embeddings.pkl").open("rb") as handle:
        return pickle.load(handle)


def _subset_cells(rna, n_cells: int | None) -> Any:
    if n_cells is None:
        return None
    return np.random.choice(rna.obs_names, size=n_cells, replace=False)


def _should_skip_condition(
    cond_results,
    *,
    n_samples: int,
    seed: int,
    n_cells: int | None,
    neighbors: int,
    graph_neighbors: int,
) -> bool:
    return (
        isinstance(cond_results, dict)
        and cond_results.get("n_decode_samples") == n_samples
        and cond_results.get("decode_seed") == seed
        and cond_results.get("n_cells") == n_cells
        and cond_results.get("neighbors") == neighbors
        and cond_results.get("graph_neighbors") == graph_neighbors
    )


def _average_replicate_decoding(
    model,
    *,
    rep_name: str,
    n_samples: int,
    seed: int,
    show_progress: bool,
) -> pd.DataFrame:
    from mern.support import average_enzyme_activity

    return average_enzyme_activity(
        [model],
        n_samples=n_samples,
        seed=seed,
        rep_names=[rep_name],
        show_progress=show_progress,
    )


def _replicate_metrics(
    model_reps: list[tuple[pd.DataFrame, dict[str, Any], str]],
    *,
    cell_idx,
    neighbors: int,
    graph_neighbors: int,
    cross: bool = False,
) -> dict[str, list]:
    from mern import support as mern_support

    met_jaccards = []
    back_jaccards = []
    graph_jaccards = []
    rxn_corrs = []
    reps = []

    pair_iter = itertools.combinations(range(len(model_reps)), 2)
    for i, j in pair_iter:
        rep1_enzyme, rep1_d, rep1 = model_reps[i]
        rep2_enzyme, rep2_d, rep2 = model_reps[j]
        metrics = _metrics_for_pair(
            rep1_enzyme,
            rep1_d,
            rep2_enzyme,
            rep2_d,
            cell_idx=cell_idx,
            neighbors=neighbors,
            graph_neighbors=graph_neighbors,
            support_module=mern_support,
            intersect_reactions=cross,
        )
        met_jaccards.append(metrics["met_jaccard"])
        back_jaccards.append(metrics["back_jaccard"])
        graph_jaccards.append(metrics["graph_jaccard"])
        rxn_corrs.append(metrics["rxn_corr"])
        reps.append((rep1, rep2))

    return {
        "met_jaccards": met_jaccards,
        "back_jaccards": back_jaccards,
        "graph_jaccards": graph_jaccards,
        "rxn_corrs": rxn_corrs,
        "reps": reps,
    }


def _metrics_for_pair(
    rep1_enzyme: pd.DataFrame,
    rep1_d: dict[str, Any],
    rep2_enzyme: pd.DataFrame,
    rep2_d: dict[str, Any],
    *,
    cell_idx,
    neighbors: int,
    graph_neighbors: int,
    support_module,
    intersect_reactions: bool,
) -> dict[str, Any]:
    if cell_idx is not None:
        rep1_enzyme = rep1_enzyme.loc[cell_idx]
        rep2_enzyme = rep2_enzyme.loc[cell_idx]
        rep1_met = rep1_d["metabolic_latent"].loc[cell_idx]
        rep2_met = rep2_d["metabolic_latent"].loc[cell_idx]
        rep1_back = rep1_d["background_latent"].loc[cell_idx]
        rep2_back = rep2_d["background_latent"].loc[cell_idx]
    else:
        rep1_met = rep1_d["metabolic_latent"]
        rep2_met = rep2_d["metabolic_latent"]
        rep1_back = rep1_d["background_latent"]
        rep2_back = rep2_d["background_latent"]

    rep1_graph = rep1_d["graph_embedding"]
    rep2_graph = rep2_d["graph_embedding"]

    if intersect_reactions:
        common_reactions = rep1_enzyme.columns.intersection(rep2_enzyme.columns)
        rep1_enzyme = rep1_enzyme[common_reactions]
        rep2_enzyme = rep2_enzyme[common_reactions]

    return {
        "met_jaccard": support_module.knn_consistency(
            rep1_met, rep2_met, k=neighbors, return_mean=False
        ),
        "back_jaccard": support_module.knn_consistency(
            rep1_back, rep2_back, k=neighbors, return_mean=False
        ),
        "graph_jaccard": support_module.knn_consistency(
            rep1_graph, rep2_graph, k=graph_neighbors, return_mean=False
        ),
        "rxn_corr": support_module.reaction_correlation(rep1_enzyme, rep2_enzyme),
    }


def _load_existing_evaluation_results(results_file: Path) -> tuple[dict, dict]:
    if not results_file.exists():
        return {}, {}

    with results_file.open("rb") as handle:
        results = pickle.load(handle)
        losses = pickle.load(handle)
    return results, losses


def run_evaluate_models_command(args: argparse.Namespace) -> int:
    if args.n_samples < 1:
        raise ValueError("--n_samples must be at least 1")

    np.random.seed(args.seed)

    print("Loading anndata and preparing dataset...")
    rna = _prepare_adata(args.adata_path, args.species)
    cell_idx = _subset_cells(rna, args.n_cells)

    results_file = args.dir / "evaluation_results.pkl"
    results, losses = _load_existing_evaluation_results(results_file)

    for condition_dir in tqdm(_condition_dirs(args.dir), disable=args.quiet):
        cond = condition_dir.name
        if cond in results and _should_skip_condition(
            results[cond],
            n_samples=args.n_samples,
            seed=args.seed,
            n_cells=args.n_cells,
            neighbors=args.neighbors,
            graph_neighbors=args.graph_neighbors,
        ):
            continue

        model_reps = []
        loss = []
        for rep_dir in _condition_dirs(condition_dir):
            try:
                model = _load_mern_model(rep_dir, rna, args.accelerator)
            except Exception as error:
                print(f"Warning: Failed to process model at {rep_dir}: {error}")
                continue

            embeddings = _read_embeddings(rep_dir)
            enzyme_act = _average_replicate_decoding(
                model,
                rep_name=rep_dir.name,
                n_samples=args.n_samples,
                seed=args.seed,
                show_progress=not args.quiet,
            )
            try:
                loss.append(embeddings["history"])
            except KeyError:
                loss.append(model.history)
            model_reps.append((enzyme_act, embeddings, rep_dir.name))
            del model

        losses[cond] = loss
        cond_metrics = _replicate_metrics(
            model_reps,
            cell_idx=cell_idx,
            neighbors=args.neighbors,
            graph_neighbors=args.graph_neighbors,
        )
        cond_metrics.update(
            {
                "n_decode_samples": args.n_samples,
                "decode_seed": args.seed,
                "n_cells": args.n_cells,
                "neighbors": args.neighbors,
                "graph_neighbors": args.graph_neighbors,
            }
        )
        results[cond] = cond_metrics

    with results_file.open("wb") as handle:
        pickle.dump(results, handle)
        pickle.dump(losses, handle)

    print(f"Done! Evaluation results saved to {results_file}")
    return 0


def _load_models_from_dir(model_dir: Path, adata, *, accelerator: str) -> list[tuple]:
    model_reps = []
    for rep_name in os.listdir(model_dir):
        rep_dir = model_dir / rep_name
        if not rep_dir.is_dir():
            continue
        try:
            model = _load_mern_model(rep_dir, adata, accelerator)
        except Exception as error:
            print(f"Warning: Failed to process model at {rep_dir}: {error}")
            continue
        embeddings = _read_embeddings(rep_dir)
        enzyme_act = model.get_decoding()["enzyme_activity"]
        model_reps.append((enzyme_act, embeddings, rep_dir.name))
        del model

    return model_reps


def _cross_output_path(dir1: Path, dir2: Path) -> Path:
    dir2_path_parts = str(dir2).rstrip("/").split(os.sep)
    if len(dir2_path_parts) >= 3:
        dir2_basename = "_".join(dir2_path_parts[-3:])
    else:
        dir2_basename = "_".join(dir2_path_parts)
    return dir1 / f"vs_{dir2_basename}.pkl"


def run_evaluate_cross_models_command(args: argparse.Namespace) -> int:
    np.random.seed(args.seed)

    print("Loading first anndata and preparing dataset...")
    rna = _prepare_adata(args.adata_path, args.species)

    if args.adata_path2 is not None:
        print("Loading second anndata and preparing dataset...")
        rna2 = _prepare_adata(args.adata_path2, args.species)
    else:
        rna2 = rna

    cell_idx = _subset_cells(rna, args.n_cells)

    dir1_models = _load_models_from_dir(args.dir1, rna, accelerator=args.accelerator)
    dir2_models = _load_models_from_dir(args.dir2, rna2, accelerator=args.accelerator)

    from mern import support as mern_support

    met_jaccards = []
    back_jaccards = []
    graph_jaccards = []
    rxn_corrs = []
    model_pairs = []

    for model1 in tqdm(dir1_models, disable=args.quiet):
        for model2 in dir2_models:
            enzyme1, d1, name1 = model1
            enzyme2, d2, name2 = model2
            metrics = _metrics_for_pair(
                enzyme1,
                d1,
                enzyme2,
                d2,
                cell_idx=cell_idx,
                neighbors=args.neighbors,
                graph_neighbors=args.graph_neighbors,
                support_module=mern_support,
                intersect_reactions=True,
            )
            met_jaccards.append(metrics["met_jaccard"])
            back_jaccards.append(metrics["back_jaccard"])
            graph_jaccards.append(metrics["graph_jaccard"])
            rxn_corrs.append(metrics["rxn_corr"])
            model_pairs.append((name1, name2))

    results = {
        "cross_evaluation": {
            "met_jaccards": met_jaccards,
            "back_jaccards": back_jaccards,
            "graph_jaccards": graph_jaccards,
            "rxn_corrs": rxn_corrs,
            "model_pairs": model_pairs,
        }
    }

    results_file = _cross_output_path(args.dir1, args.dir2)
    with results_file.open("wb") as handle:
        pickle.dump(results, handle)

    print(f"Done! Cross-evaluation results saved to {results_file}")
    return 0
