"""Create small reproducibility fixtures from the mouse intestine data."""

from __future__ import annotations

import pickle
from pathlib import Path

import anndata as ad

from mern.support import KeggKGMLMetabolicDataset


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ADATA = Path("/Users/daniellewinsohn/Desktop/NiWag/mern/data/intestine/mouse_intestine_pp.h5ad.gz")
TEST_DATA_DIR = ROOT / "tests" / "data"
ADATA_OUT = TEST_DATA_DIR / "mouse_intestine_100.h5ad"
GRAPH_OUT = TEST_DATA_DIR / "mouse_intestine_100_mern_inputs.pkl"


def main() -> None:
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(SOURCE_ADATA)[:100, :].copy()
    adata.write_h5ad(ADATA_OUT, compression="gzip")

    dataset = KeggKGMLMetabolicDataset(species="mouse", capitalize=False)
    graph = dataset.metabolic_topology(adata, self_loops=False)
    rxn_to_genes = dataset.get_rxn_genes_all()

    with GRAPH_OUT.open("wb") as handle:
        pickle.dump(
            {
                "graph": graph,
                "rxn_to_genes": rxn_to_genes,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print(f"Wrote {ADATA_OUT} with shape {adata.shape}")
    print(
        f"Wrote {GRAPH_OUT} with {graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges, and {len(rxn_to_genes)} reaction mappings"
    )


if __name__ == "__main__":
    main()
