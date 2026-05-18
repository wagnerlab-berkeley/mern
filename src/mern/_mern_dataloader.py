from torch.utils.data import DataLoader
from itertools import cycle
import networkx as nx
import numpy as np
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from scvi.data import AnnDataManager
from scvi.dataloaders import AnnDataLoader
from ._graph_dataloader import GraphDataLoader

class MERNDataLoader(DataLoader):
    """DataLoader that supports loading both cell data and graph data.

    Parameters
    ----------
    adata_manager
        :class:`~scvi.data.AnnDataManager` object that has been created via ``setup_anndata``.
    graph
        NetworkX graph or PyG Data object containing the metabolic network.
    neg_sampling_ratio
        Ratio of negative to positive edges.
    shuffle
        Whether to shuffle the data.
    batch_size
        Batch size.
    data_and_attirbutes
        Dictionary with keys representing keys in data registry (``adata_manager.data_registry``)
        and value equal to desired numpy loading type (later made into torch tensor).
        If ``None``, defaults to all registered data.
    drop_last
        If `True` and the dataset is not evenly divisible by `batch_size`, the last
        incomplete batch is dropped. If `False` and the dataset is not evenly divisible
        by `batch_size`, then the last batch will be smaller than `batch_size`.
    data_loader_kwargs
        Keyword arguments for :class:`~torch.utils.data.DataLoader`.
    distributed_sampler
        ``EXPERIMENTAL`` Whether to use :class:`~scvi.dataloaders.BatchDistributedSampler` as the
        sampler. If `True`, `sampler` must be `None`.
    """

    def __init__(   
        self,
        adata_manager: AnnDataManager,
        graph: Data,
        neg_sampling_ratio: int = 1,
        indices: list[int] | None = None,
        shuffle: bool = False,
        batch_size: int = 128,
        data_and_attributes: dict | None = None,
        drop_last: bool | int = False,
        distributed_sampler: bool = False,
        **data_loader_kwargs,
    ):
        self.adata_manager = adata_manager
        self.graph = graph
        self.neg_sampling_ratio = neg_sampling_ratio
        self._shuffle = shuffle
        self._batch_size = batch_size
        self._drop_last = drop_last
        self._data_and_attributes = data_and_attributes
        self.data_loader_kwargs = data_loader_kwargs
        self._distributed_sampler = distributed_sampler
        self.cell_indices = indices

        self.cell_loader = AnnDataLoader(
            self.adata_manager,
            indices=self.cell_indices,
            shuffle=self._shuffle,
            batch_size=self._batch_size,
            data_and_attributes=self._data_and_attributes,
            drop_last=self._drop_last,
            distributed_sampler=self._distributed_sampler,
            **self.data_loader_kwargs,
        )
        self.graph_loader = GraphDataLoader(
            self.graph,
            length=len(self.cell_loader),
            neg_sampling_ratio=self.neg_sampling_ratio,
            undirected=False,
        )
        
        data_loader_kwargs.pop("load_sparse_tensor", None)

        super().__init__(self.cell_loader, **data_loader_kwargs)

    def __len__(self):
        return len(self.cell_loader)

    def __iter__(self):
        return ({"cells": cells, "graph": graph} for cells, graph in zip(self.cell_loader, self.graph_loader))
        
        
        
        

