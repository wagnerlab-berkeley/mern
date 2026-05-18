from pathlib import Path

import anndata as ad
import pytest


TEST_DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def mouse_intestine_100_path() -> Path:
    return TEST_DATA_DIR / "mouse_intestine_100.h5ad"


@pytest.fixture
def mouse_intestine_100(mouse_intestine_100_path):
    return ad.read_h5ad(mouse_intestine_100_path)
