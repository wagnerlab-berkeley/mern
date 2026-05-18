"""Create the small mouse intestine AnnData reproducibility fixture."""

from __future__ import annotations

from pathlib import Path

import anndata as ad

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ADATA = Path(
    "/Users/daniellewinsohn/Desktop/NiWag/mern/data/intestine/mouse_intestine_pp.h5ad.gz"
)
TEST_DATA_DIR = ROOT / "tests" / "data"
ADATA_OUT = TEST_DATA_DIR / "mouse_intestine_100.h5ad"


def main() -> None:
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(SOURCE_ADATA)[:100, :].copy()
    adata.write_h5ad(ADATA_OUT, compression="gzip")

    print(f"Wrote {ADATA_OUT} with shape {adata.shape}")


if __name__ == "__main__":
    main()
