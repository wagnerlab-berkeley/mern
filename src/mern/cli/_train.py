"""Training commands for MERN."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def _add_bool_argument(
    parser: argparse.ArgumentParser,
    name: str,
    *,
    default: bool = False,
    help: str,
) -> None:
    parser.add_argument(name, action="store_true", default=default, help=help)


def add_train_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("train", help="Train MERN models")
    train_subparsers = parser.add_subparsers(dest="train_command", required=True)
    add_reps_parser(train_subparsers)


def add_reps_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "reps",
        help="Train MERN replicates and save latent representations",
        description="Train MERN replicates and save latent representations.",
    )
    parser.add_argument("--rep", type=int, default=0, help="Replicate number to use as seed")
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save the model",
    )
    parser.add_argument(
        "--data_dir", "--data-dir", type=Path, help="Directory containing the data"
    )
    parser.add_argument(
        "--adata_file", "--adata-file", type=str, help="Name of AnnData file to read"
    )
    _add_bool_argument(
        parser,
        "--all_metabolic",
        default=False,
        help="Use all metabolic genes instead of only HVGs",
    )
    parser.add_argument(
        "--all-metabolic", dest="all_metabolic", action="store_true", help=argparse.SUPPRESS
    )
    _add_bool_argument(parser, "--encode_covariates", default=False, help="Encode covariates")
    parser.add_argument(
        "--encode-covariates",
        dest="encode_covariates",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--n_metabolic_dim",
        "--n-metabolic-dim",
        type=int,
        default=25,
        help="Number of metabolic dimensions",
    )
    parser.add_argument(
        "--n_background_dim",
        "--n-background-dim",
        type=int,
        default=15,
        help="Number of background dimensions",
    )
    _add_bool_argument(
        parser,
        "--positive_met_dims",
        default=False,
        help="Force metabolic dimensions to be positive",
    )
    parser.add_argument(
        "--positive-met-dims",
        dest="positive_met_dims",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    _add_bool_argument(
        parser, "--fixed_rxn_genes", default=False, help="Fix reaction-to-genes mapping"
    )
    parser.add_argument(
        "--fixed-rxn-genes", dest="fixed_rxn_genes", action="store_true", help=argparse.SUPPRESS
    )
    _add_bool_argument(
        parser,
        "--strict_met_back_separation",
        default=True,
        help="Strictly separate metabolic and background genes",
    )
    parser.add_argument(
        "--strict-met-back-separation",
        dest="strict_met_back_separation",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no_strict_met_back_separation",
        "--no-strict-met-back-separation",
        dest="strict_met_back_separation",
        action="store_false",
        help="Do not strictly separate metabolic and background genes",
    )
    parser.add_argument(
        "--g_elbo_weight",
        "--g-elbo-weight",
        type=float,
        default=0.2,
        help="Weight for graph reconstruction loss",
    )
    parser.add_argument(
        "--graph_kl_weight",
        "--graph-kl-weight",
        type=float,
        default=0.1,
        help="Weight for graph KL divergence",
    )
    parser.add_argument(
        "--max_kl_weight",
        "--max-kl-weight",
        type=float,
        default=0.001,
        help="Maximum weight for KL divergence",
    )
    parser.add_argument(
        "--rxn_genes_weight",
        "--rxn-genes-weight",
        type=float,
        default=0.5,
        help="Weight for reaction-to-genes loss",
    )
    parser.add_argument(
        "--background_to_metabolic_weight",
        "--background-to-metabolic-weight",
        type=float,
        default=30000,
        help="Weight for background-to-metabolic loss",
    )
    parser.add_argument(
        "--species", type=str, default="mouse", help="Species to use for metabolic network"
    )
    parser.add_argument(
        "--batch_key",
        "--batch-key",
        type=str,
        default=None,
        help="Column name for batch information",
    )
    parser.add_argument(
        "--n_hidden", "--n-hidden", type=int, default=256, help="Number of hidden units"
    )
    parser.add_argument(
        "--n_layers", "--n-layers", type=int, default=2, help="Number of hidden layers"
    )
    parser.add_argument(
        "--leiden_resolution",
        "--leiden-resolution",
        type=float,
        default=0.5,
        help="Resolution for Leiden clustering",
    )
    parser.add_argument(
        "--min_cell_counts",
        "--min-cell-counts",
        type=int,
        default=-1,
        help="Minimum number of cells expressing metabolic genes",
    )
    parser.add_argument(
        "--max_epochs",
        "--max-epochs",
        type=int,
        default=1500,
        help="Maximum number of training epochs",
    )
    parser.add_argument(
        "--n_steps_kl_warmup",
        "--n-steps-kl-warmup",
        type=int,
        default=0,
        help="Number of steps to warm up KL weight",
    )
    _add_bool_argument(
        parser,
        "--separate_hvgs",
        default=True,
        help="Separate HVG selection for metabolic and non-metabolic genes",
    )
    parser.add_argument(
        "--separate-hvgs", dest="separate_hvgs", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--no_separate_hvgs",
        "--no-separate-hvgs",
        dest="separate_hvgs",
        action="store_false",
        help="Use only the highly_variable flag instead of separate metabolic/background HVGs",
    )
    _add_bool_argument(
        parser, "--rxn_genes_bias", default=True, help="Add bias term to reaction-to-genes layer"
    )
    parser.add_argument(
        "--rxn-genes-bias", dest="rxn_genes_bias", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--no_rxn_genes_bias",
        "--no-rxn-genes-bias",
        dest="rxn_genes_bias",
        action="store_false",
        help="Do not add a bias term to the reaction-to-genes layer",
    )
    _add_bool_argument(
        parser, "--fixed_graph_cell_kl", default=False, help="Fix graph cell KL divergence"
    )
    parser.add_argument(
        "--fixed-graph-cell-kl",
        dest="fixed_graph_cell_kl",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--learning_rate",
        "--learning-rate",
        type=float,
        default=2e-3,
        help="Learning rate for training",
    )
    parser.add_argument(
        "--early_stopping_patience",
        "--early-stopping-patience",
        type=int,
        default=45,
        help="Number of epochs to wait before early stopping",
    )
    _add_bool_argument(
        parser,
        "--omit_oxphos",
        default=False,
        help="Omit oxphos reactions from the metabolic network",
    )
    parser.add_argument(
        "--omit-oxphos", dest="omit_oxphos", action="store_true", help=argparse.SUPPRESS
    )
    _add_bool_argument(
        parser, "--shuffle_rxn_genes", default=False, help="Shuffle the reaction-to-genes mapping"
    )
    parser.add_argument(
        "--shuffle-rxn-genes",
        dest="shuffle_rxn_genes",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--exclude_rxn_file",
        "--exclude-rxn-file",
        type=str,
        default=None,
        help="File within data_dir containing reactions whose genes should not be included",
    )
    _add_bool_argument(
        parser, "--shuffle_graph", default=False, help="Shuffle the metabolic graph"
    )
    parser.add_argument(
        "--shuffle-graph", dest="shuffle_graph", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--shuffle_graph_seed",
        "--shuffle-graph-seed",
        type=int,
        default=8,
        help="Random seed to use if shuffling the metabolic graph",
    )
    parser.add_argument(
        "--remove_edges",
        "--remove-edges",
        type=float,
        default=None,
        help="Remove a fraction of edges from the metabolic graph randomly",
    )
    parser.add_argument(
        "--accelerator",
        default="cuda",
        help="Lightning accelerator to use for training and loading the best model",
    )
    parser.add_argument(
        "--devices", default=None, help="Lightning devices value to pass to model.train"
    )
    parser.set_defaults(func=run_reps_command)


def set_all_seeds(seed: int = 42) -> None:
    import random

    import numpy as np
    import scvi
    import torch

    scvi.settings.seed = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"All random seeds set to {seed}")


def _print_reps_args(args: argparse.Namespace) -> None:
    print("=== COMMAND LINE ARGUMENTS ===")
    for name in (
        "rep",
        "output_dir",
        "data_dir",
        "adata_file",
        "all_metabolic",
        "encode_covariates",
        "n_metabolic_dim",
        "n_background_dim",
        "positive_met_dims",
        "fixed_rxn_genes",
        "strict_met_back_separation",
        "g_elbo_weight",
        "graph_kl_weight",
        "max_kl_weight",
        "rxn_genes_weight",
        "background_to_metabolic_weight",
        "species",
        "batch_key",
        "n_hidden",
        "n_layers",
        "leiden_resolution",
        "min_cell_counts",
        "max_epochs",
        "n_steps_kl_warmup",
        "separate_hvgs",
        "rxn_genes_bias",
        "fixed_graph_cell_kl",
        "learning_rate",
        "early_stopping_patience",
        "omit_oxphos",
        "shuffle_rxn_genes",
        "exclude_rxn_file",
        "shuffle_graph",
        "shuffle_graph_seed",
        "remove_edges",
    ):
        value = getattr(args, name)
        print(f"{name}: {value} (type: {type(value)})")


def _data_path(args: argparse.Namespace) -> Path:
    if args.data_dir is None:
        raise ValueError("--data_dir is required")
    if args.adata_file is None:
        raise ValueError("--adata_file is required")
    return args.data_dir / args.adata_file


def _devices_value(devices: str | None) -> str | int | None:
    if devices is None:
        return None
    try:
        return int(devices)
    except ValueError:
        return devices


def run_reps_command(args: argparse.Namespace) -> int:
    import pickle
    import random

    import anndata as ad
    import networkx as nx
    import numpy as np
    import pandas as pd
    import scanpy as sc
    from scvi.train._callbacks import SaveCheckpoint

    import mern
    from mern import support as mern_support

    _print_reps_args(args)
    set_all_seeds(args.rep)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rna = ad.read_h5ad(_data_path(args))

    add_oxphos = not args.omit_oxphos
    dataset = mern_support.KeggKGMLMetabolicDataset(
        species=args.species, capitalize=False, add_oxphos=add_oxphos
    )
    dataset.add_module_info(rna)

    if args.separate_hvgs and args.all_metabolic:
        raise ValueError("Cannot use both all_metabolic and separate_hvg")

    if args.all_metabolic:
        rna.var["cell_counts"] = np.asarray((rna.layers["counts_RNA"] > 0).sum(axis=0)).ravel()
        features_to_use = list(
            rna.var_names[
                (
                    rna.var["highly_variable"]
                    | (
                        (rna.var["Metabolic Gene"] == "Metabolic")
                        & (rna.var["cell_counts"] > args.min_cell_counts)
                    )
                )
            ]
        )
    elif args.separate_hvgs:
        features_to_use = list(
            rna.var_names[
                rna.var["highly_variable_metabolic"] | rna.var["highly_variable_background"]
            ]
        )
    else:
        print("including only hvgs")
        features_to_use = list(rna.var_names[rna.var["highly_variable"]])

    features_to_use.sort()
    rna = rna[:, features_to_use].copy()

    met_g = dataset.metabolic_topology(rna, self_loops=False)
    dataset.add_rxn_module_info(rna, met_g)
    rxn_genes = dataset.get_rxn_genes_all()

    if args.exclude_rxn_file is not None:
        exclude_file_path = args.data_dir / args.exclude_rxn_file
        print(
            "Excluding genes for reactions listed in "
            f"{exclude_file_path} from rxn_genes (setting gene list to empty for those reactions)"
        )
        if not exclude_file_path.exists():
            raise FileNotFoundError(f"--exclude_rxn_file path does not exist: {exclude_file_path}")
        with exclude_file_path.open() as f:
            lines = f.readlines()
        exclude_reactions = {line.strip() for line in lines if line.strip()}
        affected = 0
        for rxn in exclude_reactions:
            if rxn in rxn_genes:
                rxn_genes[rxn] = []
                affected += 1
        print(f"Set gene list to empty for {affected} reactions in rxn_genes.")

    if args.shuffle_rxn_genes:
        print("Shuffling rxn_genes (reaction-to-genes mapping)...")
        keys = list(rxn_genes.keys())
        values = list(rxn_genes.values())
        random.shuffle(values)
        rxn_genes = dict(zip(keys, values))

    if args.shuffle_graph:
        print("Shuffling graph...")
        random.seed(args.shuffle_graph_seed)
        graph_relabeled = met_g.copy()
        nodes = list(graph_relabeled.nodes())
        random.shuffle(nodes)
        mapping = {old: new for old, new in zip(nodes, list(graph_relabeled.nodes()))}
        graph_relabeled = nx.relabel_nodes(graph_relabeled, mapping, copy=True)
        met_g = graph_relabeled

    if args.remove_edges is not None:
        print("Removing edges...")
        random.seed(args.shuffle_graph_seed)
        unique_pairs = sorted(set(tuple(sorted((u, v))) for u, v in met_g.edges()))
        num_remove = int(len(unique_pairs) * args.remove_edges)
        pairs_to_remove = random.sample(unique_pairs, num_remove)
        edges_to_remove = []
        for u, v in pairs_to_remove:
            edges_to_remove.append((u, v))
            edges_to_remove.append((v, u))
        met_g.remove_edges_from(edges_to_remove)

    print(rna)
    print(rna.var["Metabolic Gene"].value_counts())

    model_path = args.output_dir / f"mern_rep_{args.rep}"
    embeddings_path = model_path / "embeddings.pkl"
    if not embeddings_path.exists():
        print("Training MeRN...")
        plan_kwargs = {
            "max_kl_weight": args.max_kl_weight,
            "graph_kl_weight": args.graph_kl_weight,
            "data_elbo_weight": 1.0,
            "graph_elbo_weight": args.g_elbo_weight,
            "rxn_genes_weight": args.rxn_genes_weight,
            "background_to_metabolic_weight": args.background_to_metabolic_weight,
            "lr": args.learning_rate,
            "n_steps_kl_warmup": args.n_steps_kl_warmup,
            "n_epochs_kl_warmup": None,
        }

        checkpointing = SaveCheckpoint(
            dirpath=args.output_dir / "scvi_log" / f"mern_rep_{args.rep}",
            monitor="validation_loss",
            load_best_on_end=False,
            check_nan_gradients=False,
        )
        mern.MERN.setup_anndata(rna, layer="counts", batch_key=args.batch_key)
        model = mern.MERN(
            rna,
            met_g,
            rxn_genes,
            n_hidden=args.n_hidden,
            n_layers=args.n_layers,
            n_metabolic_dim=args.n_metabolic_dim,
            n_background_dim=args.n_background_dim,
            encode_covariates=args.encode_covariates,
            positive_met_dims=args.positive_met_dims,
            fixed_rxn_genes=args.fixed_rxn_genes,
            strict_met_back_separation=args.strict_met_back_separation,
            rxn_genes_bias=args.rxn_genes_bias,
            fixed_graph_cell_kl=args.fixed_graph_cell_kl,
        )
        train_kwargs = {}
        devices = _devices_value(args.devices)
        if devices is not None:
            train_kwargs["devices"] = devices
        model.train(
            accelerator=args.accelerator,
            max_epochs=args.max_epochs,
            early_stopping=True,
            early_stopping_monitor="validation_loss",
            early_stopping_patience=args.early_stopping_patience,
            plan_kwargs=plan_kwargs,
            callbacks=[checkpointing],
            **train_kwargs,
        )

        best_mern = mern.MERN.load(
            checkpointing.best_model_path, rna, accelerator=args.accelerator
        )

        history = model.history
        best_mern.save(model_path)

        metabolic_latent, background_latent, graph_embedding = (
            best_mern.get_latent_representation()
        )
        rna.obsm["X_mern_metabolic"] = metabolic_latent
        rna.obsm["X_mern_background"] = background_latent

        graph_embedding = pd.DataFrame(graph_embedding, index=best_mern.vertex_names_ordered)
        metabolic_latent = pd.DataFrame(metabolic_latent, index=rna.obs_names)
        background_latent = pd.DataFrame(background_latent, index=rna.obs_names)

        save_dict_before = {
            "metabolic_latent": metabolic_latent,
            "background_latent": background_latent,
            "graph_embedding": graph_embedding,
            "features_to_use": features_to_use,
            "history": history,
            "train_indices": model.train_indices,
            "validation_indices": model.validation_indices,
            "test_indices": model.test_indices,
        }

        with embeddings_path.open("wb") as f:
            pickle.dump(save_dict_before, f)

    else:
        print("Loading embeddings from file...")
        with embeddings_path.open("rb") as f:
            save_dict_before = pickle.load(f)

        rna.obsm["X_mern_metabolic"] = save_dict_before["metabolic_latent"].to_numpy()
        rna.obsm["X_mern_background"] = save_dict_before["background_latent"].to_numpy()

    if "metabolic_umap" in save_dict_before.keys():
        print("Metabolic UMAP already exists... exiting")
    else:
        print("Running clustering and UMAP...")
        sc.pp.neighbors(rna, use_rep="X_mern_metabolic", metric="euclidean")
        sc.tl.leiden(
            rna,
            key_added=f"Metabolic Clustering_{args.leiden_resolution}",
            resolution=args.leiden_resolution,
        )
        sc.tl.umap(rna)
        metabolic_umap = rna.obsm["X_umap"].copy()

        sc.pp.neighbors(rna, use_rep="X_mern_background", metric="euclidean")
        sc.tl.leiden(
            rna,
            key_added=f"Background Clustering_{args.leiden_resolution}",
            resolution=args.leiden_resolution,
        )
        sc.tl.umap(rna)
        background_umap = rna.obsm["X_umap"].copy()

        metabolic_umap = pd.DataFrame(metabolic_umap, index=rna.obs_names)
        background_umap = pd.DataFrame(background_umap, index=rna.obs_names)
        metabolic_clustering = rna.obs[f"Metabolic Clustering_{args.leiden_resolution}"].copy()
        background_clustering = rna.obs[f"Background Clustering_{args.leiden_resolution}"].copy()

        save_dict = save_dict_before
        save_dict["metabolic_umap"] = metabolic_umap
        save_dict["background_umap"] = background_umap
        save_dict["metabolic_clustering"] = metabolic_clustering
        save_dict["background_clustering"] = background_clustering

        with embeddings_path.open("wb") as f:
            pickle.dump(save_dict, f)

    return 0
