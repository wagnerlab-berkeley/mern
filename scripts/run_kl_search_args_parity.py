"""Run old/new parity using the generate_mouse_intestine_kl_search argument bundle.

This intentionally mirrors the paper training setup from:

* mern-paper/bash_scripts/generate_mouse_intestine_kl_search.sh
* mern-paper/scripts/generate_reps_smart.py

It runs on the compact 100-cell fixture so it can be used as a fast regression
guard before changing internals.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import anndata as ad
import networkx as nx
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
ADATA_PATH = ROOT / "tests" / "data" / "mouse_intestine_100.h5ad"


def _as_array_summary(value: Any) -> dict[str, Any]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    arr = np.asarray(value, dtype=float)
    return {
        "shape": list(arr.shape),
        "sum": float(arr.sum()),
        "mean": float(arr.mean()),
        "std": float(arr.std()) if arr.size > 1 else 0.0,
    }


def _history_last(history: Any) -> dict[str, float]:
    out = {}
    if history is None:
        return out
    for key in sorted(history.keys()):
        values = history[key]
        try:
            value = values.iloc[-1] if hasattr(values, "iloc") else values[-1]
            if hasattr(value, "iloc"):
                value = value.iloc[0]
            out[key] = float(np.asarray(value))
        except Exception:
            continue
    return out


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    return repr(value)


def _state_summary(model) -> dict[str, Any]:
    state = model.module.state_dict()
    out = {"total_params": 0, "state_abs_sum": 0.0}
    for tensor in state.values():
        if torch.is_tensor(tensor):
            tensor = tensor.detach().cpu().float()
            out["total_params"] += tensor.numel()
            out["state_abs_sum"] += tensor.abs().sum().item()
    return out


def _first_batch_loss(model) -> dict[str, Any]:
    dl = model._make_data_loader(model.adata, indices=np.arange(model.adata.n_obs), batch_size=32)
    batch = next(iter(dl))
    model.module.eval()
    with torch.no_grad():
        _, _, loss_output = model.module(batch)
    out = {"loss": _as_array_summary(loss_output.loss)}
    for key, value in loss_output.extra_metrics.items():
        out[key] = _as_array_summary(value)
    return out


def _kl_search_values(array_task_id: int) -> tuple[int, float, float]:
    max_kl_weights = [0.0001, 0.001, 0.005, 0.01]
    graph_kl_weights = [0.01, 0.1]
    rep = array_task_id % 15
    param_idx = array_task_id // 15
    max_kl_idx = param_idx % 4
    graph_kl_idx = param_idx // 4
    return rep, max_kl_weights[max_kl_idx], graph_kl_weights[graph_kl_idx]


def _load_support(impl: str):
    if impl == "old":
        import importlib.util
        import sys
        import types

        old_support = ROOT.parent / "mern-support" / "mern_support"
        package_name = "legacy_kl_msupport"
        package = types.ModuleType(package_name)
        package.__path__ = [str(old_support)]
        sys.modules[package_name] = package

        for name in ("config", "_addEdge"):
            path = old_support / f"{name}.py"
            spec = importlib.util.spec_from_file_location(f"{package_name}.{name}", path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[f"{package_name}.{name}"] = module
            spec.loader.exec_module(module)
            setattr(package, name, module)

        path = old_support / "_metabolic_datasets.py"
        spec = importlib.util.spec_from_file_location(f"{package_name}._metabolic_datasets", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"{package_name}._metabolic_datasets"] = module
        spec.loader.exec_module(module)
        return module

    from mern.support import _metabolic_datasets

    return _metabolic_datasets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=["old", "new"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--array-task-id", type=int, default=None)
    parser.add_argument("--rep", type=int, default=1)
    parser.add_argument("--max-kl-weight", type=float, default=0.001)
    parser.add_argument("--graph-kl-weight", type=float, default=0.1)
    parser.add_argument("--n-metabolic-dim", type=int, default=25)
    parser.add_argument("--n-background-dim", type=int, default=15)
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--all-metabolic", action="store_true", default=False)
    parser.add_argument("--encode-covariates", action="store_true", default=False)
    parser.add_argument("--positive-met-dims", action="store_true", default=False)
    parser.add_argument("--fixed-rxn-genes", action="store_true", default=False)
    parser.add_argument("--strict-met-back-separation", action="store_true", default=True)
    parser.add_argument("--no-strict-met-back-separation", dest="strict_met_back_separation", action="store_false")
    parser.add_argument("--g-elbo-weight", type=float, default=0.2)
    parser.add_argument("--rxn-genes-weight", type=float, default=0.5)
    parser.add_argument("--background-to-metabolic-weight", type=float, default=30000)
    parser.add_argument("--species", type=str, default="mouse")
    parser.add_argument("--batch-key", type=str, default=None)
    parser.add_argument("--n-hidden", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--min-cell-counts", type=int, default=-1)
    parser.add_argument("--separate-hvgs", action="store_true", default=True)
    parser.add_argument("--no-separate-hvgs", dest="separate_hvgs", action="store_false")
    parser.add_argument("--rxn-genes-bias", action="store_true", default=True)
    parser.add_argument("--no-rxn-genes-bias", dest="rxn_genes_bias", action="store_false")
    parser.add_argument("--fixed-graph-cell-kl", action="store_true", default=False)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--omit-oxphos", action="store_true", default=False)
    parser.add_argument("--shuffle-rxn-genes", action="store_true", default=False)
    parser.add_argument("--shuffle-graph", action="store_true", default=False)
    parser.add_argument("--shuffle-graph-seed", type=int, default=8)
    parser.add_argument("--remove-edges", type=float, default=None)
    args = parser.parse_args()

    if args.array_task_id is None:
        rep = args.rep
        max_kl_weight = args.max_kl_weight
        graph_kl_weight = args.graph_kl_weight
    else:
        rep, max_kl_weight, graph_kl_weight = _kl_search_values(args.array_task_id)
    random.seed(rep)
    np.random.seed(rep)
    torch.manual_seed(rep)
    torch.set_num_threads(1)

    if args.impl == "old":
        from scvi.external.mern import MERN
    else:
        from mern import MERN
    import scvi

    scvi.settings.seed = rep

    adata = ad.read_h5ad(ADATA_PATH)

    support_datasets = _load_support(args.impl)
    dataset = support_datasets.KeggKGMLMetabolicDataset(
        species=args.species,
        capitalize=False,
        add_oxphos=not args.omit_oxphos,
    )
    dataset.add_module_info(adata)

    if args.separate_hvgs and args.all_metabolic:
        raise ValueError("Cannot use both all_metabolic and separate_hvgs")
    if args.all_metabolic:
        count_layer = "counts_RNA" if "counts_RNA" in adata.layers else "counts"
        adata.var["cell_counts"] = np.asarray((adata.layers[count_layer] > 0).sum(axis=0)).ravel()
        features_to_use = list(
            adata.var_names[
                adata.var["highly_variable"]
                | (
                    (adata.var["Metabolic Gene"] == "Metabolic")
                    & (adata.var["cell_counts"] > args.min_cell_counts)
                )
            ]
        )
    elif args.separate_hvgs:
        features_to_use = list(
            adata.var_names[
                adata.var["highly_variable_metabolic"] | adata.var["highly_variable_background"]
            ]
        )
    else:
        features_to_use = list(adata.var_names[adata.var["highly_variable"]])
    features_to_use.sort()
    adata = adata[:, features_to_use].copy()

    graph = dataset.metabolic_topology(adata, self_loops=False)
    dataset.add_rxn_module_info(adata, graph)
    rxn_to_genes = dataset.get_rxn_genes_all()

    if args.shuffle_rxn_genes:
        keys = list(rxn_to_genes.keys())
        values = list(rxn_to_genes.values())
        random.shuffle(values)
        rxn_to_genes = dict(zip(keys, values))

    if args.shuffle_graph:
        random.seed(args.shuffle_graph_seed)
        relabeled = graph.copy()
        nodes = list(relabeled.nodes())
        random.shuffle(nodes)
        mapping = {old: new for old, new in zip(nodes, list(relabeled.nodes()))}
        graph = nx.relabel_nodes(relabeled, mapping, copy=True)

    if args.remove_edges is not None:
        random.seed(args.shuffle_graph_seed)
        unique_pairs = sorted(set(tuple(sorted((u, v))) for u, v in graph.edges()))
        num_remove = int(len(unique_pairs) * args.remove_edges)
        pairs_to_remove = random.sample(unique_pairs, num_remove)
        edges_to_remove = []
        for u, v in pairs_to_remove:
            edges_to_remove.append((u, v))
            edges_to_remove.append((v, u))
        graph.remove_edges_from(edges_to_remove)

    plan_kwargs = {
        "max_kl_weight": max_kl_weight,
        "graph_kl_weight": graph_kl_weight,
        "data_elbo_weight": 1.0,
        "graph_elbo_weight": args.g_elbo_weight,
        "rxn_genes_weight": args.rxn_genes_weight,
        "background_to_metabolic_weight": args.background_to_metabolic_weight,
        "lr": args.learning_rate,
        "n_steps_kl_warmup": 0,
        "n_epochs_kl_warmup": None,
    }

    MERN.setup_anndata(adata, layer="counts", batch_key=args.batch_key)
    model = MERN(
        adata,
        graph,
        rxn_to_genes,
        n_hidden=args.n_hidden,
        n_layers=args.n_layers,
        n_metabolic_dim=args.n_metabolic_dim,
        n_background_dim=args.n_background_dim,
        encode_covariates=args.encode_covariates,
        positive_met_dims=args.positive_met_dims,
        fixed_rxn_genes=args.fixed_rxn_genes,
        strict_met_back_separation=args.strict_met_back_separation,
        rxn_genes_bias=args.rxn_genes_bias,
        fixed_graph_cell_kl=args.fixed_graph_cell_kl,
    )

    result = {
        "impl": args.impl,
        "scvi_version": scvi.__version__,
        "array_task_id": args.array_task_id,
        "rep": rep,
        "max_kl_weight": max_kl_weight,
        "graph_kl_weight": graph_kl_weight,
        "model_args": {
            "n_hidden": args.n_hidden,
            "n_layers": args.n_layers,
            "n_metabolic_dim": args.n_metabolic_dim,
            "n_background_dim": args.n_background_dim,
            "encode_covariates": args.encode_covariates,
            "positive_met_dims": args.positive_met_dims,
            "fixed_rxn_genes": args.fixed_rxn_genes,
            "strict_met_back_separation": args.strict_met_back_separation,
            "rxn_genes_bias": args.rxn_genes_bias,
            "fixed_graph_cell_kl": args.fixed_graph_cell_kl,
        },
        "plan_kwargs": plan_kwargs,
        "paper_script_args": {
            "all_metabolic": args.all_metabolic,
            "species": args.species,
            "batch_key": args.batch_key,
            "min_cell_counts": args.min_cell_counts,
            "separate_hvgs": args.separate_hvgs,
            "omit_oxphos": args.omit_oxphos,
            "shuffle_rxn_genes": args.shuffle_rxn_genes,
            "shuffle_graph": args.shuffle_graph,
            "shuffle_graph_seed": args.shuffle_graph_seed,
            "remove_edges": args.remove_edges,
        },
        "adata_shape": list(adata.shape),
        "features_to_use_count": len(features_to_use),
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges(),
        "initial_state": _state_summary(model),
        "initial_first_batch": _first_batch_loss(model),
    }

    if not args.skip_train:
        model.train(
            accelerator="cpu",
            devices=1,
            max_epochs=args.max_epochs,
            early_stopping=False,
            train_size=0.8,
            validation_size=0.2,
            shuffle_set_split=False,
            batch_size=32,
            plan_kwargs=plan_kwargs,
            enable_checkpointing=False,
            logger=False,
            enable_model_summary=False,
        )

        result["history_last"] = _history_last(model.history)
        result["trained_state"] = _state_summary(model)
        result["trained_first_batch"] = _first_batch_loss(model)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_safe(result), indent=2, sort_keys=True))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
