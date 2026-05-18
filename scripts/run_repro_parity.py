"""Run a deterministic old/new MERN parity payload.

Use this script under either the old scvi-tools-mern environment or the new
merged-package environment. It saves compact numeric summaries for comparison.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
ADATA_PATH = ROOT / "tests" / "data" / "mouse_intestine_100.h5ad"


def _as_float(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return float(np.asarray(value))


def _numeric_summary(value: Any) -> dict[str, Any]:
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
            if hasattr(values, "iloc"):
                value = values.iloc[-1]
            else:
                value = values[-1]
            if hasattr(value, "iloc"):
                value = value.iloc[0]
            out[key] = _as_float(value)
        except Exception:
            continue
    return out


def _state_summary(model) -> dict[str, Any]:
    state = model.module.state_dict()
    summary = {}
    total = 0
    sum_abs = 0.0
    for name, tensor in state.items():
        if not torch.is_tensor(tensor):
            continue
        tensor = tensor.detach().cpu().float()
        total += tensor.numel()
        sum_abs += tensor.abs().sum().item()
        if name in {
            "z_encoder.encoder.fc_layers.Layer 0.0.weight",
            "decoder.rxn_gene_layer.linear.weight",
            "graph_encoder.vrepr",
        }:
            summary[name] = {
                "shape": list(tensor.shape),
                "sum": tensor.sum().item(),
                "mean": tensor.mean().item(),
                "std": tensor.std().item() if tensor.numel() > 1 else 0.0,
            }
    summary["total_params"] = total
    summary["state_abs_sum"] = sum_abs
    return summary


def _first_batch_loss(model) -> dict[str, float]:
    dl = model._make_data_loader(model.adata, indices=np.arange(model.adata.n_obs), batch_size=16)
    batch = next(iter(dl))
    model.module.eval()
    with torch.no_grad():
        _, _, loss_output = model.module(batch)
    out = {"loss": _numeric_summary(loss_output.loss)}
    for key, value in loss_output.extra_metrics.items():
        out[key] = _numeric_summary(value)
    return out


def _load_support(impl: str):
    if impl == "old":
        import importlib.util
        import sys
        import types

        old_support = ROOT.parent / "mern-support" / "mern_support"
        package_name = "legacy_repro_msupport"
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
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=1)
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(1)

    if args.impl == "old":
        from scvi.external.mern import MERN
    else:
        from mern import MERN

    import scvi

    scvi.settings.seed = 0

    adata = ad.read_h5ad(ADATA_PATH)
    support_datasets = _load_support(args.impl)
    dataset = support_datasets.KeggKGMLMetabolicDataset(
        species="mouse",
        capitalize=False,
        add_oxphos=True,
    )
    dataset.add_module_info(adata)
    graph = dataset.metabolic_topology(adata, self_loops=False)
    rxn_to_genes = dataset.get_rxn_genes_all()

    MERN.setup_anndata(adata, layer="counts", batch_key=None)
    model = MERN(
        adata,
        graph,
        rxn_to_genes,
        n_hidden=32,
        n_layers=1,
        n_metabolic_dim=5,
        n_background_dim=3,
        dropout_rate=0.0,
        encode_covariates=False,
        strict_met_back_separation=True,
        rxn_genes_bias=True,
    )

    result = {
        "impl": args.impl,
        "scvi_version": scvi.__version__,
        "adata_shape": list(adata.shape),
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges(),
        "initial_state": _state_summary(model),
        "initial_first_batch": _first_batch_loss(model),
    }

    if args.train:
        model.train(
            max_epochs=args.max_epochs,
            accelerator="cpu",
            devices=1,
            train_size=0.8,
            validation_size=0.2,
            shuffle_set_split=False,
            batch_size=32,
            plan_kwargs={
                "n_epochs_kl_warmup": None,
                "n_steps_kl_warmup": 0,
            },
            enable_checkpointing=False,
            logger=False,
            enable_model_summary=False,
        )
        result["history_last"] = _history_last(model.history)
        result["trained_state"] = _state_summary(model)
        result["trained_first_batch"] = _first_batch_loss(model)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
