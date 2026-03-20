"""Configuration settings for the MERN package."""

from pathlib import Path


PACKAGE_DIR = Path(__file__).parent
KEGG_DIR = str(PACKAGE_DIR / "data" / "kegg")
ANNDATA_KEY = "anndata"
