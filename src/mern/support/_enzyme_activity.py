"""Helpers for aggregating MERN enzyme activity predictions."""

from collections.abc import Iterable, Sequence
import os
import random

import numpy as np
import pandas as pd
from tqdm import tqdm


def _set_all_seeds(seed: int = 42) -> None:
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


def _rep_seed_id(rep_name, fallback: int) -> int:
    try:
        return int(str(rep_name).split("_")[-1])
    except ValueError:
        return fallback


def average_enzyme_activity(
    models: Iterable,
    adata=None,
    n_samples: int = 8,
    seed: int = 8,
    rep_names: Sequence | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Average predicted enzyme activity across MERN model replicates.

    This mirrors the ``average_enzyme_activity.py`` scripts used by
    ``mern-paper`` and ``mern-folate``. Each model is decoded ``n_samples``
    times, with per-decode seeds derived as ``seed + rep_id * 1_000 +
    sample_idx``. When ``rep_names`` is omitted, the model's list index is used
    as ``rep_id``. If ``adata`` is provided, it is passed to each model's
    ``get_decoding`` call.
    """
    if n_samples < 1:
        raise ValueError("--n_samples must be at least 1")

    if rep_names is not None:
        rep_names = list(rep_names)

    enzyme_sum = None
    decode_count = 0
    column_names = None
    cell_index = None
    model_count = 0

    model_iter = tqdm(models, desc="Processing models", disable=not show_progress)
    for rep_idx, model in enumerate(model_iter):
        if rep_names is not None:
            try:
                rep_name = rep_names[rep_idx]
            except IndexError as error:
                raise ValueError("rep_names must have the same length as models.") from error
        else:
            rep_name = getattr(model, "_mern_average_rep_name", rep_idx)
        rep_id = _rep_seed_id(rep_name, rep_idx)

        for sample_idx in tqdm(
            range(n_samples),
            desc=f"Decoding {rep_name}",
            leave=False,
            disable=not show_progress,
        ):
            decode_seed = seed + rep_id * 1_000 + sample_idx
            _set_all_seeds(decode_seed)
            if n_samples > 1 and show_progress:
                tqdm.write(
                    f"{rep_name}: decoding sample {sample_idx + 1}/{n_samples} "
                    f"with seed {decode_seed}"
                )

            enzyme_act = model.get_decoding(adata)["enzyme_activity"]

            if enzyme_sum is None:
                cell_index = enzyme_act.index
                column_names = enzyme_act.columns
                enzyme_sum = np.zeros(enzyme_act.shape, dtype=np.float64)
            elif not enzyme_act.columns.equals(column_names):
                enzyme_act = enzyme_act.loc[:, column_names]

            enzyme_sum += enzyme_act.values
            decode_count += 1
        model_count += 1

    if rep_names is not None and len(rep_names) != model_count:
        raise ValueError("rep_names must have the same length as models.")

    if enzyme_sum is None:
        raise ValueError("No models were provided.")

    enzyme_avg = enzyme_sum / decode_count
    return pd.DataFrame(enzyme_avg, index=cell_index, columns=column_names)
