"""Run old/new parity using the generate_mouse_intestine_kl_search argument bundle."""

from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path
from typing import Any

import anndata as ad
import networkx as nx
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
ADATA_PATH = ROOT / "tests" / "data" / "mouse_intestine_100.h5ad"
INPUTS_PATH = ROOT / "tests" / "data" / "mouse_intestine_100_mern_inputs.pkl"


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=["old", "new"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--array-task-id", type=int, default=16)
    parser.add_argument("--n-metabolic-dim", type=int, default=25)
    parser.add_argument("--n-background-dim", type=int, default=15)
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

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
    with INPUTS_PATH.open("rb") as handle:
        inputs = pickle.load(handle)

    # Mimic --separate_hvgs from generate_mouse_intestine_kl_search.sh.
    features_to_use = list(
        adata.var_names[
            adata.var["highly_variable_metabolic"] | adata.var["highly_variable_background"]
        ]
    )
    features_to_use.sort()
    adata = adata[:, features_to_use].copy()

    graph = inputs["graph"].copy()
    rxn_to_genes = inputs["rxn_to_genes"]
    graph = graph.subgraph(
        [
            node
            for node in graph.nodes
            if any(gene in adata.var_names for gene in rxn_to_genes.get(node, []))
        ]
    ).copy()
    nx.set_edge_attributes(graph, 1.0, "weight")
    nx.set_edge_attributes(graph, 1, "sign")

    plan_kwargs = {
        "max_kl_weight": max_kl_weight,
        "graph_kl_weight": graph_kl_weight,
        "data_elbo_weight": 1.0,
        "graph_elbo_weight": 0.2,
        "rxn_genes_weight": 0.5,
        "background_to_metabolic_weight": 30000,
        "lr": 2e-3,
        "n_steps_kl_warmup": 0,
        "n_epochs_kl_warmup": None,
    }

    MERN.setup_anndata(adata, layer="counts", batch_key=None)
    model = MERN(
        adata,
        graph,
        rxn_to_genes,
        n_hidden=256,
        n_layers=2,
        n_metabolic_dim=args.n_metabolic_dim,
        n_background_dim=args.n_background_dim,
        encode_covariates=False,
        positive_met_dims=False,
        fixed_rxn_genes=False,
        strict_met_back_separation=True,
        rxn_genes_bias=True,
        fixed_graph_cell_kl=False,
    )

    result = {
        "impl": args.impl,
        "scvi_version": scvi.__version__,
        "array_task_id": args.array_task_id,
        "rep": rep,
        "max_kl_weight": max_kl_weight,
        "graph_kl_weight": graph_kl_weight,
        "model_args": {
            "n_hidden": 256,
            "n_layers": 2,
            "n_metabolic_dim": args.n_metabolic_dim,
            "n_background_dim": args.n_background_dim,
            "encode_covariates": False,
            "positive_met_dims": False,
            "fixed_rxn_genes": False,
            "strict_met_back_separation": True,
            "rxn_genes_bias": True,
            "fixed_graph_cell_kl": False,
        },
        "plan_kwargs": plan_kwargs,
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
