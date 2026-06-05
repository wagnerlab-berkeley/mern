import numpy as np
import pandas as pd
import pytest

from mern.support import average_enzyme_activity
import mern.support._enzyme_activity as enzyme_activity_module


class _FakeMERN:
    def __init__(self, columns):
        self.columns = columns
        self.decoding_adatas = []

    def get_decoding(self, adata=None):
        self.decoding_adatas.append(adata)
        values = np.random.random((1, len(self.columns)))
        return {
            "enzyme_activity": pd.DataFrame(values, index=["cell_1"], columns=self.columns),
        }


def test_average_enzyme_activity_matches_script_seed_schedule(monkeypatch):
    seeds = []

    def fake_set_all_seeds(seed):
        seeds.append(seed)
        np.random.seed(seed)

    monkeypatch.setattr(enzyme_activity_module, "_set_all_seeds", fake_set_all_seeds)
    models = [_FakeMERN(["r1", "r2"]), _FakeMERN(["r2", "r1"])]

    averaged = average_enzyme_activity(models, n_samples=2, seed=8, show_progress=False)

    expected_rows = []
    for decode_seed, columns in [
        (8, ["r1", "r2"]),
        (9, ["r1", "r2"]),
        (1008, ["r2", "r1"]),
        (1009, ["r2", "r1"]),
    ]:
        np.random.seed(decode_seed)
        expected_rows.append(
            pd.DataFrame(np.random.random((1, 2)), index=["cell_1"], columns=columns).loc[
                :, ["r1", "r2"]
            ]
        )
    expected = sum(df.values for df in expected_rows) / len(expected_rows)

    assert seeds == [8, 9, 1008, 1009]
    assert list(averaged.index) == ["cell_1"]
    assert list(averaged.columns) == ["r1", "r2"]
    assert np.allclose(averaged.values, expected)


def test_average_enzyme_activity_uses_rep_name_suffix_for_seeds(monkeypatch):
    seeds = []

    def fake_set_all_seeds(seed):
        seeds.append(seed)
        np.random.seed(seed)

    monkeypatch.setattr(enzyme_activity_module, "_set_all_seeds", fake_set_all_seeds)

    average_enzyme_activity(
        [_FakeMERN(["r1"]), _FakeMERN(["r1"])],
        n_samples=1,
        seed=8,
        rep_names=["rep_3", "not_numeric"],
        show_progress=False,
    )

    assert seeds == [3008, 1008]


def test_average_enzyme_activity_forwards_adata_to_models():
    adata = object()
    models = [_FakeMERN(["r1"]), _FakeMERN(["r1"])]

    average_enzyme_activity(models, adata=adata, n_samples=2, show_progress=False)

    assert models[0].decoding_adatas == [adata, adata]
    assert models[1].decoding_adatas == [adata, adata]


def test_average_enzyme_activity_validates_inputs():
    with pytest.raises(ValueError, match="n_samples"):
        average_enzyme_activity([_FakeMERN(["r1"])], n_samples=0, show_progress=False)

    with pytest.raises(ValueError, match="No models"):
        average_enzyme_activity([], show_progress=False)

    with pytest.raises(ValueError, match="rep_names"):
        average_enzyme_activity([_FakeMERN(["r1"])], rep_names=[], show_progress=False)
