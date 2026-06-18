"""Average enzyme activity command for MERN."""

from __future__ import annotations

import argparse
from pathlib import Path


def add_average_enzyme_activity_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "average-enzyme-activity",
        aliases=["average_enzyme_activity"],
        help="Average predicted enzyme activity across MERN model replicates",
        description="Average predicted enzyme activity across MERN model replicates.",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        required=True,
        help="Directory containing model replicate directories",
    )
    parser.add_argument(
        "--adata_path",
        "--adata-path",
        type=Path,
        required=True,
        help="Path to AnnData file",
    )
    parser.add_argument(
        "--output_path",
        "--output-path",
        type=Path,
        default=None,
        help="Path to save average enzyme activity",
    )
    parser.add_argument(
        "--species",
        type=str,
        default="mouse",
        help="Species for KeggKGMLMetabolicDataset",
    )
    parser.add_argument(
        "--n_samples",
        "--n-samples",
        type=int,
        default=8,
        help="Number of independent decoding samples to average per model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=8,
        help="Base seed used to derive independent per-decode seeds",
    )
    parser.add_argument(
        "--accelerator",
        default="cpu",
        help="Accelerator to use when loading models",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Disable progress bars and per-decode progress messages",
    )
    parser.set_defaults(func=run_average_enzyme_activity_command)


def _prepare_adata(adata_path: Path, species: str):
    import anndata as ad

    from mern import support as mern_support

    print("Loading anndata and preparing dataset...")
    rna = ad.read_h5ad(adata_path)
    dataset = mern_support.KeggKGMLMetabolicDataset(species=species, capitalize=False)
    dataset.add_module_info(rna)

    features_to_use = list(
        rna.var_names[rna.var["highly_variable_metabolic"] | rna.var["highly_variable_background"]]
    )
    features_to_use.sort()
    rna = rna[:, features_to_use].copy()

    met_g = dataset.metabolic_topology(rna, self_loops=False)
    dataset.add_rxn_module_info(rna, met_g)
    return rna


def _model_replicate_dirs(models_dir: Path) -> list[Path]:
    return sorted(path for path in models_dir.iterdir() if path.is_dir())


def run_average_enzyme_activity_command(args: argparse.Namespace) -> int:
    import mern
    from mern.support import average_enzyme_activity

    if args.n_samples < 1:
        raise ValueError("--n_samples must be at least 1")

    output_path = args.output_path
    if output_path is None:
        output_path = args.dir / "average_enzyme_activity.csv.gz"

    rna = _prepare_adata(args.adata_path, args.species)
    model_dirs = _model_replicate_dirs(args.dir)

    print(f"Using base seed {args.seed} to derive independent seeds for each decoding sample.")
    print(f"Processing models from {args.dir} with {args.n_samples} sample(s) per model...")

    def load_models():
        for model_dir in model_dirs:
            try:
                model = mern.MERN.load(model_dir, rna, accelerator=args.accelerator)
                model._mern_average_rep_name = model_dir.name
                yield model
            except Exception as error:
                print(f"Warning: Failed to process model at {model_dir}: {error}")

    enzyme_avg_df = average_enzyme_activity(
        load_models(),
        n_samples=args.n_samples,
        seed=args.seed,
        show_progress=not args.quiet,
    )

    print(f"Saving average enzyme activity to {output_path}...")
    compression = "gzip" if str(output_path).endswith(".gz") else None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    enzyme_avg_df.to_csv(output_path, index=True, compression=compression)

    print(f"Done! Average enzyme activity saved to {output_path}")
    return 0
