import importlib
import inspect
import sys


def test_public_imports_and_version():
    import mern

    assert mern.__version__ == "1.0.1"
    assert mern.MERN.__name__ == "MERN"
    assert mern.MERNModule.__name__ == "MERNModule"
    assert mern.MERNDataLoader.__name__ == "MERNDataLoader"
    assert mern.GraphDataLoader.__name__ == "GraphDataLoader"


def test_support_import_does_not_eagerly_import_plots():
    sys.modules.pop("mern.support._plots", None)
    support = importlib.import_module("mern.support")

    assert "mern.support._plots" not in sys.modules
    assert support.__version__ == "1.0.1"


def test_support_plot_exports_lazy_load_when_accessed():
    sys.modules.pop("mern.support._plots", None)
    support = importlib.import_module("mern.support")

    assert support.training_plot.__name__ == "training_plot"
    assert "mern.support._plots" in sys.modules


def test_mern_init_signature_contains_expected_arguments():
    from mern import MERN

    params = inspect.signature(MERN.__init__).parameters
    expected = {
        "adata",
        "graph",
        "rxn_to_genes",
        "n_hidden",
        "n_layers",
        "n_metabolic_dim",
        "n_background_dim",
        "strict_met_back_separation",
        "rxn_genes_bias",
    }
    assert expected.issubset(params)
    assert "kwargs" in params
    assert params["n_background_dim"].default == 10
    assert params["strict_met_back_separation"].default is True
    assert params["rxn_genes_bias"].default is True
