"""
DDP LLM-oriented export helpers.

Callers provide:
  * **Cell × DDP** activity (e.g. mean of member reaction activities; aggregation lives elsewhere).
  * **DDP definitions**: map ``DDP_k`` -> decode-matrix / graph reaction column names.
  * **Static graph context**: KEGG rxn–rxn graph + ``Reaction Info`` frame (from
    ``KeggKGMLMetabolicDataset.add_rxn_module_info``) + optional ``rxn_genes``.
  * **Biology context**: ``AnnData.obs`` columns (categorical / numeric), optionally restricted
    by a **boolean mask** (e.g. cell-type subset). Summaries describe **how each DDP’s activity
    varies along those axes** (category means; numeric axes via quantile bins)—not statistical
    comparisons between context columns.
  * **DDP subsetting**: keep only DDPs that touch a user **pathway** list
    (``ddp_ids_matching_pathways``) and/or **reaction** list (``ddp_ids_matching_reactions``),
    then ``filter_ddp_definitions`` / ``filter_ddp_activity`` before building static + obs blocks.

This module does not compute DDP means from raw enzyme activity.

For a single-call export, use :func:`export_ddp_llm_bundle`.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd

SCHEMA_VERSION = "1.5"

# Abstract wording for third-party LLM upload (no architecture / method details).
DEFAULT_SCORE_INTERPRETATION = (
    "Each cell has **numeric scores for metabolic reactions** (one value per reaction id). "
    "Scores summarize gene-expression evidence for that reaction in that cell; they are **not** "
    "raw sequencing counts and are only meant for **relative comparison** across cells and along "
    "the biological axes provided below."
)

DEFAULT_DDP_INTERPRETATION = (
    "**DDPs** (data-driven pathways; labels like DDP_0, DDP_1, …) are predefined **groups of "
    "reactions**. Each DDP value for a cell is the **mean** of its member reaction scores. "
    "Interpret a DDP using reaction ids, reaction descriptions, and pathway map ids in the "
    "record—finer than a single pathway map, coarser than one reaction. (These are not KEGG "
    "modules M#####; KEGG module ids may appear separately in reaction metadata.)"
)


def _json_sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, np.generic):
        return _json_sanitize(obj.item())
    if isinstance(obj, float):
        return None if not np.isfinite(obj) else obj
    return obj


def _split_decode_rxns(decode_key: str) -> List[str]:
    return [t for t in str(decode_key).strip().split() if t]


def _reaction_info_for_decode_key(
    reaction_info: pd.DataFrame,
    graph: nx.DiGraph,
    decode_key: str,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Returns a dict with pathway/module fields and the list of pathway ids (unique, sorted).
    Uses the graph node row if present; otherwise merges rows for whitespace-split tokens.
    """
    if decode_key in reaction_info.index:
        row = reaction_info.loc[decode_key]
        pws = _parse_semicolon_ids(str(row.get("Pathways", "")))
        pwn = _parse_semicolon_ids(str(row.get("Pathway Names", "")))
        return (
            {
                "decode_matrix_key": decode_key,
                "rxn_display_names": str(row.get("Rxn Name", "")),
                "pathway_ids": pws,
                "pathway_names": pwn,
                "modules": str(row.get("Modules", "")),
            },
            pws,
        )

    tokens = _split_decode_rxns(decode_key)
    pws_acc: Set[str] = set()
    names_acc: List[str] = []
    disp: List[str] = []
    mods: List[str] = []
    for t in tokens:
        if t not in reaction_info.index:
            continue
        row = reaction_info.loc[t]
        disp.append(str(row.get("Rxn Name", "")))
        mods.append(str(row.get("Modules", "")))
        for pid in _parse_semicolon_ids(str(row.get("Pathways", ""))):
            pws_acc.add(pid)
        for pn in _filter_empty(str(row.get("Pathway Names", "")).split(";")):
            names_acc.append(pn.strip())
    pws_sorted = sorted(pws_acc)
    return (
        {
            "decode_matrix_key": decode_key,
            "constituent_rxns_in_graph": [t for t in tokens if t in graph.nodes],
            "rxn_display_names": "; ".join(disp) if disp else "",
            "pathway_ids": pws_sorted,
            "pathway_names": names_acc,
            "modules": "; ".join(mods) if mods else "",
        },
        pws_sorted,
    )


def _parse_semicolon_ids(s: str) -> List[str]:
    return sorted({_x.strip() for _x in s.split(";") if _x.strip()})


def _filter_empty(xs: Iterable[str]) -> List[str]:
    return [x for x in xs if x]


def _expanded_pathway_ids(raw: Sequence[str]) -> Set[str]:
    """
    Normalize user pathway ids for matching ``Reaction Info`` entries (often ``rn#####``).
    If the user passes a species map id (``mmu#####``, ``hsa#####``), also add the ``rn#####`` form.
    """
    out: Set[str] = set()
    for p in raw:
        s = str(p).strip().lower()
        if not s:
            continue
        out.add(s)
        m = re.match(r"^(mmu|hsa|eco|dme)([0-9]{5})$", s)
        if m:
            out.add(f"rn{m.group(2)}")
    return out


def ddp_ids_matching_pathways(
    ddp_definitions: Mapping[str, Sequence[str]],
    reaction_info: pd.DataFrame,
    graph: nx.DiGraph,
    pathway_ids: Sequence[str],
) -> Set[str]:
    """
    DDPs where **at least one** member reaction carries any of the given KEGG pathway map ids
    in ``Reaction Info`` (semicolon-separated ``Pathways`` field).
    """
    required = _expanded_pathway_ids(pathway_ids)
    if not required:
        return set()
    keep: Set[str] = set()
    for ddp_id, rxn_keys in ddp_definitions.items():
        for rk in rxn_keys:
            _, pws = _reaction_info_for_decode_key(reaction_info, graph, rk)
            pnorm = {str(x).strip().lower() for x in pws}
            if required & pnorm:
                keep.add(ddp_id)
                break
    return keep


def ddp_ids_matching_reactions(
    ddp_definitions: Mapping[str, Sequence[str]],
    reaction_ids: Sequence[str],
) -> Set[str]:
    """
    DDPs where **at least one** member decode-matrix key equals or whitespace-splits to a
    reaction id in ``reaction_ids`` (e.g. ``rn:R00200``).
    """
    req: Set[str] = set()
    for r in reaction_ids:
        s = str(r).strip()
        if not s:
            continue
        req.add(s)
        if re.fullmatch(r"[Rr][0-9]{5}", s):
            req.add("rn:" + s)
        elif re.fullmatch(r"[0-9]{5}", s):
            req.add("rn:R" + s)
    keep: Set[str] = set()
    for ddp_id, rxn_keys in ddp_definitions.items():
        for rk in rxn_keys:
            if rk in req:
                keep.add(ddp_id)
                break
            if set(_split_decode_rxns(rk)) & req:
                keep.add(ddp_id)
                break
    return keep


def filter_ddp_definitions(
    ddp_definitions: Mapping[str, Sequence[str]],
    keep_ddp_ids: Set[str],
) -> Dict[str, List[str]]:
    """Subset to DDP ids; preserves insertion order of original mapping for stability."""
    return {k: list(ddp_definitions[k]) for k in ddp_definitions if k in keep_ddp_ids}


def filter_ddp_activity(ddp_activity: pd.DataFrame, keep_ddp_ids: Set[str]) -> pd.DataFrame:
    cols = [c for c in ddp_activity.columns if c in keep_ddp_ids]
    return ddp_activity[cols].copy()


def genes_for_decode_keys(
    rxn_genes: Mapping[str, Sequence[str]],
    decode_keys: Sequence[str],
    var_names: Optional[Sequence[str]] = None,
) -> List[str]:
    genes: Set[str] = set()
    for key in decode_keys:
        for t in _split_decode_rxns(key):
            genes.update(rxn_genes.get(t, []))
    if var_names is not None:
        vs = set(var_names)
        return sorted(genes & vs)
    return sorted(genes)


def build_node_to_ddps(ddp_definitions: Mapping[str, Sequence[str]]) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = defaultdict(set)
    for ddp_id, rxn_keys in ddp_definitions.items():
        for k in rxn_keys:
            for t in _split_decode_rxns(k):
                out[t].add(ddp_id)
            out[k].add(ddp_id)
    return dict(out)


def neighbor_ddps_for_graph(
    graph: nx.DiGraph,
    ddp_definitions: Mapping[str, Sequence[str]],
) -> Dict[str, List[str]]:
    """
    DDP A is adjacent to B if some graph node in A has a directed edge to/from
    some graph node in B (metabolic connectivity at the rxn–rxn graph level).
    """
    node_to_ddps = build_node_to_ddps(ddp_definitions)
    neighbors: Dict[str, Set[str]] = {d: set() for d in ddp_definitions}

    for u, v in graph.edges():
        dus = node_to_ddps.get(u)
        dvs = node_to_ddps.get(v)
        if not dus or not dvs:
            continue
        for du in dus:
            for dv in dvs:
                if du == dv:
                    continue
                neighbors[du].add(dv)
                neighbors[dv].add(du)

    return {d: sorted(neighbors[d]) for d in ddp_definitions}


def build_ddp_static_metadata(
    graph: nx.DiGraph,
    reaction_info: pd.DataFrame,
    ddp_definitions: Mapping[str, Sequence[str]],
    rxn_genes: Optional[Mapping[str, Sequence[str]]] = None,
    var_names: Optional[Sequence[str]] = None,
    *,
    include_genes: bool = False,
) -> Dict[str, Any]:
    """
    Static, dataset-agnostic DDP descriptions for LLM consumption.

    Parameters
    ----------
    graph
        Directed rxn–rxn graph from ``KeggKGMLMetabolicDataset.metabolic_topology``.
    reaction_info
        ``adata.uns['Reaction Info']`` after ``add_rxn_module_info``.
    ddp_definitions
        ``DDP_k`` -> sequence of decode-matrix reaction keys (graph node ids).
    rxn_genes
        Maps single ``rn:…`` id to gene symbols (e.g. ``dataset.rxn_genes``). Used only if
        ``include_genes`` is True.
    var_names
        If set, genes are intersected with model / AnnData var names.
    include_genes
        If False (default), **omit** ``genes_in_scope`` from each DDP so LLM consumers focus on
        reactions and pathways rather than gene symbols.
    """
    nbr = neighbor_ddps_for_graph(graph, ddp_definitions)
    ddps_out: List[Dict[str, Any]] = []

    for ddp_id in sorted(ddp_definitions.keys(), key=_ddp_sort_key):
        rxn_keys = list(ddp_definitions[ddp_id])
        pathway_union: Set[str] = set()
        rxn_entries: List[Dict[str, Any]] = []
        for rk in rxn_keys:
            info_row, pws = _reaction_info_for_decode_key(reaction_info, graph, rk)
            rxn_entries.append(info_row)
            pathway_union.update(pws)

        row: Dict[str, Any] = {
            "ddp_id": ddp_id,
            "n_member_reactions": len(rxn_keys),
            "member_reactions": rxn_keys,
            "reactions_detail": rxn_entries,
            "pathway_ids_union": sorted(pathway_union),
            "neighbor_ddps_graph": nbr.get(ddp_id, []),
        }
        if include_genes:
            if rxn_genes is not None:
                row["genes_in_scope"] = genes_for_decode_keys(
                    rxn_genes, rxn_keys, var_names=var_names
                )
            else:
                row["genes_in_scope"] = []

        ddps_out.append(row)

    return {"schema_version": SCHEMA_VERSION, "kind": "ddp_static_metadata", "ddps": ddps_out}


def _ddp_sort_key(x: str) -> Tuple[int, str]:
    if x.startswith("DDP_"):
        tail = x[4:]
        try:
            return (0, f"{int(tail):06d}")
        except ValueError:
            return (1, x)
    return (1, x)


def _obs_subset_mask(adata, obs_subset_mask: Optional[np.ndarray]) -> np.ndarray:
    if obs_subset_mask is None:
        return np.ones(adata.n_obs, dtype=bool)
    m = np.asarray(obs_subset_mask, dtype=bool)
    if m.shape[0] != adata.n_obs:
        raise ValueError(
            f"obs_subset_mask length {m.shape[0]} != adata.n_obs {adata.n_obs} (same obs order)."
        )
    return m


def build_context_guide(
    *,
    user_biological_context: str = "",
    score_interpretation: str = DEFAULT_SCORE_INTERPRETATION,
    ddp_interpretation: str = DEFAULT_DDP_INTERPRETATION,
    interpretation_note: str = "",
) -> Dict[str, str]:
    """Neutral context for LLM consumers (default avoids method/architecture detail)."""
    out: Dict[str, str] = {
        "score_interpretation": score_interpretation,
        "ddp_interpretation": ddp_interpretation,
        "dataset_biological_context": user_biological_context.strip(),
    }
    if interpretation_note.strip():
        out["interpretation_note"] = interpretation_note.strip()
    return out


def slim_provenance_for_bundle_export(
    full_provenance: Mapping[str, Any],
    *,
    include_internal_paths: bool,
) -> Dict[str, Any]:
    """
    Drop filesystem paths and internal experiment ids from provenance embedded in JSON bundles.

    Full provenance should be kept locally (e.g. ``run_meta.json``) for reproducibility.
    """
    if include_internal_paths:
        return dict(full_provenance)
    keys_keep = (
        "cells",
        "ddp_aggregation",
        "cell_subset_rule",
        "obs_context_categorical",
        "obs_context_numeric",
        "numeric_axis_n_bins",
        "static_includes_gene_symbols",
        "export_profile",
    )
    out: Dict[str, Any] = {k: full_provenance[k] for k in keys_keep if k in full_provenance}
    if "ddp_filter" in full_provenance:
        out["ddp_filter"] = full_provenance["ddp_filter"]
    return out


def _numeric_axis_quantile_summary(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_bins: int,
    min_cells_per_bin: int,
) -> Dict[str, Any]:
    """
    How DDP activity (y) varies along numeric axis x: mean y per quantile bin of x.
    """
    m = np.isfinite(x) & np.isfinite(y)
    xv = x[m].astype(np.float64)
    yv = y[m].astype(np.float64)
    n = int(xv.size)
    if n < min_cells_per_bin:
        return {
            "kind": "numeric_axis",
            "included": False,
            "n_cells_used": n,
            "reason": "too_few_finite_cells",
        }
    nq = min(int(n_bins), max(2, n // max(min_cells_per_bin, 1)))
    if nq < 2:
        return {
            "kind": "numeric_axis",
            "included": False,
            "n_cells_used": n,
            "reason": "not_enough_cells_for_two_bins",
        }
    x_series = pd.Series(xv)
    try:
        q = pd.qcut(x_series, q=nq, duplicates="drop")
    except ValueError:
        return {
            "kind": "numeric_axis",
            "included": False,
            "n_cells_used": n,
            "reason": "quantile_binning_failed",
        }
    means: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for interval in q.cat.categories:
        sel = (q == interval).to_numpy()
        nb = int(sel.sum())
        lab = str(interval)
        counts[lab] = nb
        if nb < min_cells_per_bin:
            means[lab] = float("nan")
        else:
            means[lab] = float(np.mean(yv[sel]))
    peak = max(means, key=lambda k: (-np.inf if not np.isfinite(means[k]) else means[k]))
    edges: List[float] = []
    for inter in q.cat.categories:
        edges.extend([float(inter.left), float(inter.right)])
    return {
        "kind": "numeric_axis",
        "included": True,
        "summary": (
            "Mean DDP activity by quantile bin along this numeric obs (roughly equal cell counts "
            "per bin when possible)."
        ),
        "n_bins_requested": int(n_bins),
        "n_bins_effective": int(len(q.cat.categories)),
        "bin_edges_sorted_unique": sorted(set(edges)),
        "mean_activity_by_bin": means,
        "n_cells_by_bin": counts,
        "peak_bin_by_mean_activity": peak,
        "n_cells_used": n,
    }


def summarize_ddp_obs_context(
    adata,
    ddp_activity: pd.DataFrame,
    obs_categorical: Sequence[str],
    obs_numeric: Sequence[str] = (),
    min_cells_per_group: int = 5,
    obs_subset_mask: Optional[np.ndarray] = None,
    numeric_axis_n_bins: int = 5,
) -> Dict[str, Any]:
    """
    Summarize how each DDP’s activity varies along ``obs`` columns: per-category means for
    categoricals; for numeric columns, mean activity by quantile bin along that axis.

    ``ddp_activity.index`` is aligned to ``adata.obs_names`` (reindexed with optional fill).
    ``obs_subset_mask``: optional boolean mask (same length as ``adata``) restricting which cells
    contribute (e.g. a cell-type subset). If ``None``, all cells are used.
    """
    missing_cat = [c for c in obs_categorical if c not in adata.obs.columns]
    if missing_cat:
        raise KeyError(f"Missing categorical obs columns: {missing_cat}")
    missing_num = [c for c in obs_numeric if c not in adata.obs.columns]
    if missing_num:
        raise KeyError(f"Missing numeric obs columns: {missing_num}")

    m = _obs_subset_mask(adata, obs_subset_mask)
    ad_s = adata[m].copy()
    X = ddp_activity.reindex(adata.obs_names).loc[m]
    ddps = list(X.columns)
    out_ddps: List[Dict[str, Any]] = []

    for ddp_col in ddps:
        y = pd.to_numeric(X[ddp_col], errors="coerce").to_numpy(dtype=np.float64)
        block: Dict[str, Any] = {"ddp_id": ddp_col, "by_obs": {}}

        for col in obs_categorical:
            grp = ad_s.obs[col].astype(str)
            means: Dict[str, float] = {}
            counts: Dict[str, int] = {}
            for level in sorted(grp.unique()):
                sel = grp.to_numpy() == level
                n = int(sel.sum())
                counts[level] = n
                if n < min_cells_per_group:
                    means[level] = float("nan")
                else:
                    yy = y[sel]
                    fin = np.isfinite(yy)
                    means[level] = float(np.nanmean(yy[fin])) if fin.any() else float("nan")
            peak = max(means, key=lambda k: (-np.inf if not np.isfinite(means[k]) else means[k]))
            block["by_obs"][col] = {
                "kind": "categorical",
                "mean_activity_by_category": means,
                "n_cells_by_category": counts,
                "peak_category_by_mean_activity": peak,
            }

        for col in obs_numeric:
            x = pd.to_numeric(ad_s.obs[col], errors="coerce").to_numpy(dtype=np.float64)
            block["by_obs"][col] = _numeric_axis_quantile_summary(
                x,
                y,
                n_bins=numeric_axis_n_bins,
                min_cells_per_bin=min_cells_per_group,
            )

        out_ddps.append(block)

    return {"schema_version": SCHEMA_VERSION, "kind": "ddp_obs_context", "ddps": out_ddps}


def merge_ddp_llm_bundle(
    static_block: Mapping[str, Any],
    obs_block: Mapping[str, Any],
    provenance: Mapping[str, Any],
    *,
    context_guide: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    static_by_id = {d["ddp_id"]: d for d in static_block.get("ddps", [])}
    obs_by_id = {d["ddp_id"]: d for d in obs_block.get("ddps", [])}
    keys = sorted(set(static_by_id) | set(obs_by_id), key=_ddp_sort_key)
    merged: List[Dict[str, Any]] = []
    for k in keys:
        row: Dict[str, Any] = {"ddp_id": k}
        if k in static_by_id:
            row["static"] = static_by_id[k]
        if k in obs_by_id:
            row["obs_context"] = obs_by_id[k]
        merged.append(row)
    bundle: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "ddp_llm_bundle",
        "context_guide": dict(context_guide) if context_guide is not None else {},
        "provenance": dict(provenance),
        "ddps": merged,
    }
    return bundle


def write_ddp_llm_bundle_json(bundle: Mapping[str, Any], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = _json_sanitize(dict(bundle))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(safe, f, indent=2, allow_nan=False)


def write_ddp_llm_bundle_jsonl(bundle: Mapping[str, Any], path: Union[str, Path]) -> None:
    """One JSON object per DDP line (easy chunking for RAG). Repeats bundle-level context."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prov = bundle.get("provenance", {})
    cg = bundle.get("context_guide", {})
    with open(path, "w", encoding="utf-8") as f:
        for row in bundle.get("ddps", []):
            rec = {
                "schema_version": bundle.get("schema_version"),
                "context_guide": cg,
                "provenance": prov,
                **row,
            }
            f.write(json.dumps(_json_sanitize(rec), allow_nan=False) + "\n")


def _ddp_pair_edges_from_graph(
    graph: nx.DiGraph,
    ddp_definitions: Mapping[str, Sequence[str]],
    *,
    max_example_rxn_edges_per_pair: int = 3,
) -> List[Dict[str, Any]]:
    """
    Build a compact DDP-to-DDP edge list induced by the reaction graph.

    This is intentionally "structural only": it records which DDPs are connected by at least
    one directed reaction–reaction edge, optionally with a few example reaction edges.
    """
    node_to_ddps = build_node_to_ddps(ddp_definitions)
    pair_to_count: Dict[Tuple[str, str], int] = defaultdict(int)
    pair_to_examples: Dict[Tuple[str, str], List[Tuple[str, str]]] = defaultdict(list)

    for u, v in graph.edges():
        dus = node_to_ddps.get(u)
        dvs = node_to_ddps.get(v)
        if not dus or not dvs:
            continue
        for du in dus:
            for dv in dvs:
                if du == dv:
                    continue
                key = (du, dv)
                pair_to_count[key] += 1
                ex = pair_to_examples[key]
                if len(ex) < int(max_example_rxn_edges_per_pair):
                    ex.append((u, v))

    edges_out: List[Dict[str, Any]] = []
    for (a, b), n in sorted(
        pair_to_count.items(),
        key=lambda kv: (_ddp_sort_key(kv[0][0]), _ddp_sort_key(kv[0][1])),
    ):
        edges_out.append(
            {
                "ddp_from": a,
                "ddp_to": b,
                "n_rxn_edges": int(n),
                "example_rxn_edges": [
                    {"from": u, "to": v} for (u, v) in pair_to_examples[(a, b)]
                ],
            }
        )
    return edges_out


def export_ddp_context_simple(
    *,
    graph: nx.DiGraph,
    reaction_info: pd.DataFrame,
    ddp_definitions: Mapping[str, Sequence[str]],
    ddp_ids: Optional[Sequence[str]] = None,
    context: str = "",
    output_path: Optional[Union[str, Path]] = None,
    max_example_rxn_edges_per_pair: int = 3,
    score_interpretation: str = DEFAULT_SCORE_INTERPRETATION,
    ddp_interpretation: str = DEFAULT_DDP_INTERPRETATION,
    interpretation_note: str = "",
) -> Dict[str, Any]:
    """
    Export a **simple, static-only** DDP bundle for LLM / RAG use.

    What this includes:
    - A light `context_guide` (just the provided free-text context; no biological calculations)
    - Per-DDP info with member reactions and reaction metadata from `reaction_info`
    - How DDPs connect to each other based on the reaction graph (DDP-level edge list)

    What this intentionally omits:
    - Any `obs`-based summaries (category means, numeric binning, etc.)
    - Any inferred biological interpretation beyond the raw metadata in `reaction_info`

    Parameters
    ----------
    graph, reaction_info, ddp_definitions
        Precomputed topology + reaction metadata + DDP membership.
    ddp_ids
        Optional subset of DDP ids to export. If None, exports all keys in `ddp_definitions`.
    context
        Free-text dataset/context note to embed for downstream consumers.
    output_path
        If provided, writes a single JSON file to this path.
    max_example_rxn_edges_per_pair
        Include up to this many example reaction edges per directed DDP-to-DDP edge.
    score_interpretation, ddp_interpretation
        Passed through to `context_guide` so consumers know what reaction scores and DDPs are.
    interpretation_note
        Optional extra note for `context_guide`. If empty, a default note about omitted gene
        lists is included (this export does not include per-DDP gene symbols).
    """
    if ddp_ids is None:
        defs = {k: list(ddp_definitions[k]) for k in ddp_definitions}
    else:
        requested = [str(x) for x in ddp_ids]
        missing = [d for d in requested if d not in ddp_definitions]
        if missing:
            raise KeyError(f"Requested DDP ids not found in ddp_definitions: {missing}")
        # Preserve caller-provided ordering for predictability.
        defs = {k: list(ddp_definitions[k]) for k in requested}

    static = build_ddp_static_metadata(
        graph=graph,
        reaction_info=reaction_info,
        ddp_definitions=defs,
        rxn_genes=None,
        var_names=None,
        include_genes=False,
    )

    ddps_list = list(static.get("ddps", []))
    neighbor_map = {d["ddp_id"]: d.get("neighbor_ddps_graph", []) for d in ddps_list}

    if interpretation_note.strip():
        inote = interpretation_note.strip()
    else:
        inote = (
            "Per-DDP gene symbol lists are omitted. Prefer reaction ids (member_reactions), "
            "reaction descriptions (reactions_detail), and pathway map identifiers "
            "(pathway_ids_union)."
        )

    context_guide = build_context_guide(
        user_biological_context=context,
        score_interpretation=score_interpretation,
        ddp_interpretation=ddp_interpretation,
        interpretation_note=inote,
    )

    bundle: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "ddp_context_simple",
        "context_guide": context_guide,
        "ddps": ddps_list,
        "connections": {
            "ddp_neighbor_list": neighbor_map,
            "ddp_graph_edges_directed": _ddp_pair_edges_from_graph(
                graph,
                defs,
                max_example_rxn_edges_per_pair=max_example_rxn_edges_per_pair,
            ),
        },
    }

    if output_path is not None:
        write_ddp_llm_bundle_json(bundle, output_path)

    return bundle


def _split_obs_columns(
    adata,
    obs_columns: Optional[Sequence[str]],
) -> Tuple[List[str], List[str]]:
    """Route ``obs`` columns into categorical vs numeric lists for summarization."""
    if not obs_columns:
        return [], []
    cat: List[str] = []
    num: List[str] = []
    for col in obs_columns:
        if col not in adata.obs.columns:
            raise KeyError(f"obs column {col!r} not in adata.obs")
        s = adata.obs[col]
        if pd.api.types.is_bool_dtype(s):
            cat.append(col)
        elif pd.api.types.is_numeric_dtype(s):
            num.append(col)
        else:
            cat.append(col)
    return cat, num


def _require_dataset_api(dataset: Any) -> None:
    for name in ("add_module_info", "metabolic_topology", "add_rxn_module_info"):
        if not callable(getattr(dataset, name, None)):
            raise TypeError(
                f"dataset must provide a callable {name!r} (e.g. KeggKGMLMetabolicDataset)."
            )
    if not hasattr(dataset, "rxn_genes"):
        raise TypeError("dataset must have attribute 'rxn_genes'.")


def export_ddp_llm_bundle(
    adata,
    dataset: Any,
    ddp_definitions: Mapping[str, Sequence[str]],
    ddp_activity: pd.DataFrame,
    *,
    biological_context: str = "",
    obs_columns: Optional[Sequence[str]] = None,
    output_prefix: Optional[Union[str, Path]] = None,
    include_genes: bool = False,
    interpretation_note: str = "",
    export_profile: str = "ddp_llm_export",
    ddp_aggregation: str = "mean_over_member_reactions",
    cell_subset_rule: str = "",
    min_cells_per_group: int = 5,
    numeric_axis_n_bins: int = 5,
    obs_subset_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Build a full DDP LLM bundle in one call.

    Runs ``dataset.add_module_info``, ``metabolic_topology``, and ``add_rxn_module_info`` on
    ``adata`` (same pattern as the reference export script), then static metadata, obs
    summaries, merge, and optional JSON/JSONL writes.

    Parameters
    ----------
    adata
        Cells in export; ``var_names`` used when ``include_genes`` is True.
    dataset
        Typically ``KeggKGMLMetabolicDataset`` with the methods above and ``rxn_genes``.
    ddp_definitions
        Map ``DDP_k`` -> member reaction ids (decode-matrix / graph node names).
    ddp_activity
        DataFrame indexed like ``adata.obs_names``, columns = DDP ids.
    biological_context
        Short free text stored under ``context_guide.dataset_biological_context``.
    obs_columns
        ``obs`` column names to summarize. Split automatically into categorical vs numeric
        columns (bools and non-numeric dtypes → categorical). If omitted or empty, obs
        summaries are omitted (static block only in that part of the bundle).
    output_prefix
        If set, writes ``{output_prefix}.json`` and ``{output_prefix}.jsonl``.
    include_genes
        If True, include per-DDP ``genes_in_scope`` in static metadata.
    interpretation_note
        Optional extra note for ``context_guide``. If empty and ``include_genes`` is False,
        a default note about omitted gene lists is added.
    export_profile
        Short label stored in provenance.
    ddp_aggregation
        Provenance string describing how ``ddp_activity`` was built (for consumers only).
    cell_subset_rule
        Human-readable description of cell filters; provenance only.
    min_cells_per_group, numeric_axis_n_bins, obs_subset_mask
        Passed to :func:`summarize_ddp_obs_context`.

    Returns
    -------
    dict
        The ``ddp_llm_bundle`` object (same schema as :func:`merge_ddp_llm_bundle`).
    """
    _require_dataset_api(dataset)

    dataset.add_module_info(adata)
    graph = dataset.metabolic_topology(adata, self_loops=False)
    dataset.add_rxn_module_info(adata, graph)
    reaction_info = adata.uns["Reaction Info"]

    static = build_ddp_static_metadata(
        graph=graph,
        reaction_info=reaction_info,
        ddp_definitions=ddp_definitions,
        rxn_genes=dataset.rxn_genes,
        var_names=list(adata.var_names),
        include_genes=include_genes,
    )

    obs_cat, obs_num = _split_obs_columns(adata, obs_columns)
    obs_block = summarize_ddp_obs_context(
        adata,
        ddp_activity,
        obs_categorical=obs_cat,
        obs_numeric=obs_num,
        min_cells_per_group=min_cells_per_group,
        obs_subset_mask=obs_subset_mask,
        numeric_axis_n_bins=numeric_axis_n_bins,
    )

    if interpretation_note.strip():
        inote = interpretation_note.strip()
    elif not include_genes:
        inote = (
            "Per-DDP gene symbol lists are omitted when present. Prefer reaction ids "
            "(member_reactions), reaction descriptions (reactions_detail), and pathway map "
            "identifiers (pathway_ids_union)."
        )
    else:
        inote = ""

    context_guide = build_context_guide(
        user_biological_context=biological_context,
        interpretation_note=inote,
    )

    prov_full: Dict[str, Any] = {
        "cells": int(adata.n_obs),
        "ddp_aggregation": ddp_aggregation,
        "cell_subset_rule": cell_subset_rule or "(not specified)",
        "obs_context_categorical": obs_cat,
        "obs_context_numeric": obs_num,
        "numeric_axis_n_bins": int(numeric_axis_n_bins),
        "static_includes_gene_symbols": bool(include_genes),
        "export_profile": export_profile,
    }
    prov_bundle = slim_provenance_for_bundle_export(
        prov_full,
        include_internal_paths=False,
    )

    bundle = merge_ddp_llm_bundle(
        static,
        obs_block,
        prov_bundle,
        context_guide=context_guide,
    )

    if output_prefix is not None:
        p = Path(output_prefix)
        write_ddp_llm_bundle_json(bundle, p.with_suffix(".json"))
        write_ddp_llm_bundle_jsonl(bundle, p.with_suffix(".jsonl"))

    return bundle
