from mern.cli._main import build_parser


def _parse_train_args(*extra_args):
    return build_parser().parse_args(
        [
            "train",
            "reps",
            "--output_dir",
            "/tmp/output",
            "--data_dir",
            "/tmp/data",
            "--adata_file",
            "data.h5ad",
            *extra_args,
        ]
    )


def test_train_counts_layer_default_is_unchanged():
    assert _parse_train_args().counts_layer == "counts"


def test_train_counts_layer_accepts_custom_value_and_hyphenated_alias():
    assert _parse_train_args("--counts_layer", "raw_counts").counts_layer == "raw_counts"
    assert _parse_train_args("--counts-layer", "umi_counts").counts_layer == "umi_counts"
