import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling

from ._constants import GRAPH_REGISTRY_KEYS

class GraphDataLoader(DataLoader):
    """DataLoader for graph data. Returns entire graph + new negative samples at each iteration.

    Parameters
    ----------
    graph
        NetworkX graph or PyG Data object containing the metabolic network.
    length
        Length of the graph data loader.
    neg_sampling_ratio
        Ratio of negative to positive edges
    """
    def __init__(self, graph, length, neg_sampling_ratio, undirected=True, **kwargs):
        self.graph = graph
        self.n_neg_samples = neg_sampling_ratio * graph.edge_index.size(1)
        self.undirected = undirected
        self.length = length
        super().__init__(self, **kwargs)

    def __len__(self):
        return self.length
    
    def __iter__(self):
        for _ in range(self.length):
            # Generate new negative samples for each iteration
            neg_sample = negative_sampling(
                self.graph.edge_index, 
                num_neg_samples=self.n_neg_samples, 
                force_undirected=self.undirected
            )
            
            # Combine original edges with new negative samples
            edge_index = torch.cat((self.graph.edge_index.clone().detach(), neg_sample), dim=1)
            
            # Create attributes for negative samples
            neg_edge_weight = torch.zeros(neg_sample.size(1), 1, dtype=self.graph.edge_attr.dtype)
            neg_edge_sign = torch.ones(neg_sample.size(1), 1, dtype=self.graph.edge_attr.dtype) * 1
            neg_edge_attr = torch.cat((neg_edge_weight, neg_edge_sign), dim=1)
            
            # Combine original edge attributes with negative sample attributes
            edge_attr = torch.cat((self.graph.edge_attr.clone().detach(), neg_edge_attr), dim=0)
            
            # Return dictionary with graph registry keys
            yield {
                GRAPH_REGISTRY_KEYS.EIDX_KEY: edge_index,
                GRAPH_REGISTRY_KEYS.EWT_KEY: edge_attr[:, 0],  # Edge weights
                GRAPH_REGISTRY_KEYS.ESGN_KEY: edge_attr[:, 1],  # Edge signs
            }