from pathlib import Path
import pickle

import anndata as ad
import pytest


TEST_DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def mouse_intestine_100_path() -> Path:
    return TEST_DATA_DIR / "mouse_intestine_100.h5ad"


@pytest.fixture(scope="session")
def mouse_intestine_100_mern_inputs_path() -> Path:
    return TEST_DATA_DIR / "mouse_intestine_100_mern_inputs.pkl"


@pytest.fixture
def mouse_intestine_100(mouse_intestine_100_path):
    return ad.read_h5ad(mouse_intestine_100_path)


@pytest.fixture(scope="session")
def mouse_intestine_100_mern_inputs(mouse_intestine_100_mern_inputs_path):
    with mouse_intestine_100_mern_inputs_path.open("rb") as handle:
        return pickle.load(handle)
