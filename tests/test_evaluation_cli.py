import pickle
from argparse import Namespace

import pandas as pd

from mern import support as mern_support
from mern.cli import _evaluation
from mern.cli._main import main


class _FakeAdata:
    obs_names = pd.Index(["cell_1", "cell_2"])


class _FakeModel:
    def __init__(self, enzyme_activity):
        self.enzyme_activity = enzyme_activity
        self.history = {"fallback": [1.0]}

    def get_decoding(self, adata=None):
        return {"enzyme_activity": self.enzyme_activity.copy()}


def _embedding_payload(history=None):
    cells = ["cell_1", "cell_2"]
    graph_nodes = ["rxn_1", "rxn_2"]
    return {
        "metabolic_latent": pd.DataFrame([[1.0, 0.0], [0.0, 1.0]], index=cells),
        "background_latent": pd.DataFrame([[0.5, 0.0], [0.0, 0.5]], index=cells),
        "graph_embedding": pd.DataFrame([[1.0, 1.0], [2.0, 2.0]], index=graph_nodes),
        **({} if history is None else {"history": history}),
    }


def _write_embeddings(model_dir, history=None):
    model_dir.mkdir(parents=True)
    with (model_dir / "embeddings.pkl").open("wb") as handle:
        pickle.dump(_embedding_payload(history=history), handle)


def test_evaluate_models_cli_writes_script_compatible_pickle(tmp_path, monkeypatch):
    model_root = tmp_path / "models"
    _write_embeddings(model_root / "condition_a" / "rep_0", history={"loss": [0.1]})
    _write_embeddings(model_root / "condition_a" / "rep_1")

    enzyme_by_rep = {
        "rep_0": pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=["cell_1", "cell_2"], columns=["r1", "r2"]),
        "rep_1": pd.DataFrame([[2.0, 4.0], [6.0, 8.0]], index=["cell_1", "cell_2"], columns=["r1", "r2"]),
    }

    monkeypatch.setattr(_evaluation, "_prepare_adata", lambda adata_path, species: _FakeAdata())
    monkeypatch.setattr(
        _evaluation,
        "_load_mern_model",
        lambda model_dir, adata, accelerator: _FakeModel(enzyme_by_rep[model_dir.name]),
    )
    monkeypatch.setattr(
        mern_support,
        "knn_consistency",
        lambda emb1, emb2, k, return_mean: [f"k={k}", len(emb1), len(emb2)],
    )
    monkeypatch.setattr(
        mern_support,
        "reaction_correlation",
        lambda rep1, rep2: pd.Series([0.25, 0.5], index=rep1.columns),
    )

    exit_code = main(
        [
            "evaluate-models",
            "--dir",
            str(model_root),
            "--adata_path",
            str(tmp_path / "data.h5ad"),
            "--n_samples",
            "1",
            "--quiet",
        ]
    )

    assert exit_code == 0
    with (model_root / "evaluation_results.pkl").open("rb") as handle:
        results = pickle.load(handle)
        losses = pickle.load(handle)

    condition = results["condition_a"]
    assert condition["reps"] == [("rep_0", "rep_1")]
    assert condition["met_jaccards"] == [["k=100", 2, 2]]
    assert condition["graph_jaccards"] == [["k=25", 2, 2]]
    assert condition["rxn_corrs"][0].to_dict() == {"r1": 0.25, "r2": 0.5}
    assert condition["n_decode_samples"] == 1
    assert condition["decode_seed"] == 8
    assert condition["n_cells"] is None
    assert losses["condition_a"] == [{"loss": [0.1]}, {"fallback": [1.0]}]


def test_evaluate_models_cli_skips_existing_matching_config(tmp_path, monkeypatch):
    model_root = tmp_path / "models"
    _write_embeddings(model_root / "condition_a" / "rep_0")

    existing_result = {
        "condition_a": {
            "n_decode_samples": 8,
            "decode_seed": 8,
            "n_cells": None,
            "neighbors": 100,
            "graph_neighbors": 25,
            "reps": [("old", "old")],
        }
    }
    existing_losses = {"condition_a": ["already_done"]}
    with (model_root / "evaluation_results.pkl").open("wb") as handle:
        pickle.dump(existing_result, handle)
        pickle.dump(existing_losses, handle)

    monkeypatch.setattr(_evaluation, "_prepare_adata", lambda adata_path, species: _FakeAdata())

    def fail_if_loaded(*args, **kwargs):
        raise AssertionError("matching existing conditions should be skipped")

    monkeypatch.setattr(_evaluation, "_load_mern_model", fail_if_loaded)

    assert (
        main(
            [
                "evaluate-models",
                "--dir",
                str(model_root),
                "--adata_path",
                str(tmp_path / "data.h5ad"),
                "--quiet",
            ]
        )
        == 0
    )

    with (model_root / "evaluation_results.pkl").open("rb") as handle:
        results = pickle.load(handle)
        losses = pickle.load(handle)

    assert results == existing_result
    assert losses == existing_losses


def test_evaluate_cross_models_saves_vs_file_and_intersects_reactions(tmp_path, monkeypatch):
    dir1 = tmp_path / "group1" / "reps"
    dir2 = tmp_path / "group2" / "nested" / "reps"
    _write_embeddings(dir1 / "rep_a")
    _write_embeddings(dir2 / "rep_b")

    enzymes = {
        "rep_a": pd.DataFrame(
            [[1.0, 2.0], [3.0, 4.0]],
            index=["cell_1", "cell_2"],
            columns=["shared", "left_only"],
        ),
        "rep_b": pd.DataFrame(
            [[1.0, 5.0], [3.0, 6.0]],
            index=["cell_1", "cell_2"],
            columns=["shared", "right_only"],
        ),
    }
    seen_reaction_columns = []

    monkeypatch.setattr(_evaluation, "_prepare_adata", lambda adata_path, species: _FakeAdata())
    monkeypatch.setattr(
        _evaluation,
        "_load_mern_model",
        lambda model_dir, adata, accelerator: _FakeModel(enzymes[model_dir.name]),
    )
    monkeypatch.setattr(
        mern_support,
        "knn_consistency",
        lambda emb1, emb2, k, return_mean: [k],
    )

    def fake_reaction_correlation(rep1, rep2):
        seen_reaction_columns.append((list(rep1.columns), list(rep2.columns)))
        return pd.Series([1.0], index=rep1.columns)

    monkeypatch.setattr(mern_support, "reaction_correlation", fake_reaction_correlation)

    args = Namespace(
        dir1=dir1,
        dir2=dir2,
        adata_path=tmp_path / "data1.h5ad",
        adata_path2=None,
        neighbors=100,
        graph_neighbors=25,
        n_cells=None,
        seed=8,
        species="mouse",
        accelerator="cpu",
        quiet=True,
    )

    assert _evaluation.run_evaluate_cross_models_command(args) == 0
    output_path = dir1 / "vs_group2_nested_reps.pkl"
    with output_path.open("rb") as handle:
        results = pickle.load(handle)

    cross = results["cross_evaluation"]
    assert cross["model_pairs"] == [("rep_a", "rep_b")]
    assert cross["rxn_corrs"][0].to_dict() == {"shared": 1.0}
    assert seen_reaction_columns == [(["shared"], ["shared"])]
