import networkx as nx
import numpy as np
import pandas as pd
import sklearn.linear_model
from anndata import AnnData
import pickle
import re
import itertools
from Bio import KEGG
from Bio.KEGG import REST
import sklearn.metrics
from tqdm import tqdm
import sklearn
from scipy.stats import wilcoxon, mannwhitneyu, ranksums, spearmanr
from statsmodels.stats.multitest import multipletests
from collections import defaultdict
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestNeighbors


def latent_auc(
        rna: AnnData,
        label_key: str,
        positive_label: str,
        obsm_key: str,
)-> pd.Series:
    """
    Calculates auc for each latent factor explaining a label.

    Parameters
    ----------
    rna
        includes data with labels and obsm of latent factor
    label_key
        key for label to use as target
    positive_label
        label to use as positive classification
    obsm_key
        where latent factors are stored

    Returns
    -------
    aucs
        aucs for each latent dimension
    """

    try:
        latent_embedding = rna.obsm[obsm_key]
    except KeyError: 
        raise ValueError('Latent key not valid.')

    try:
        labels = np.array(rna.obs[label_key])
    except KeyError: 
        raise ValueError('Label key not valid.')

    labels[labels == positive_label] = 1
    labels[labels != 1] = 0
    labels = labels.astype('int')

    if labels.sum() == 0: raise ValueError('Your positive label is most likely wrong.')

    aucs = []
    for dim in range(latent_embedding.shape[1]):
        aucs.append(sklearn.metrics.roc_auc_score(labels, latent_embedding[:,dim]))

    return pd.Series(aucs)

def latent_regression(
        rna: AnnData,
        label_key: str,
        positive_label: str,
        obsm_key: str,
) -> pd.Series:
    """
    Calculates logistic regression using latent space to explain a cell label

    Parameters
    ----------
    rna
        includes data with labels and obsm of latent factor
    label_key
        key for label to use as target
    positive_label
        label to use as positive classification
    obsm_key
        where latent factors are stored

    Returns
    ---------
    coefs
        logistic regression coefficients
    """
    try:
        latent_embedding = rna.obsm[obsm_key]
    except KeyError: 
        raise ValueError('Latent key not valid.')

    try:
        labels = np.array(rna.obs[label_key])
    except KeyError: 
        raise ValueError('Label key not valid.')

    # scale latent embedding so coefficients are comparable
    latent_embedding = (latent_embedding - latent_embedding.mean(axis=0))/ latent_embedding.std(axis=0)
    print(latent_embedding.mean(axis=0))
    print(latent_embedding.std(axis=0))

    labels[labels == positive_label] = 1
    labels[labels != 1] = 0
    labels = labels.astype('int')

    if labels.sum() == 0: raise ValueError('Your positive label is most likely wrong.')

    log_classifier = sklearn.linear_model.LogisticRegression(random_state=0).fit(latent_embedding, labels)

    coefs = log_classifier.coef_.flatten()

    return pd.Series(coefs)

def get_gene_sets(
        rna: AnnData,
        module: str,
):
    """
    gets gene sets for specified modules

    Parameters
    ---------
    rna
        anndata object with module info
    module
        module to get genes for

    Returns
    --------
    genes
        all genes included in module
    """
    if "Gene Modules" not in rna.var:
        raise ValueError("Please run add_module_info first!")
    
    module_info = rna.var['Gene Modules']
    genes = []
    for gene in rna.var_names:
        modules = module_info.loc[gene].split(',')
        if module in modules:
            genes.append(gene)

    return genes

def cohens_d(x, y):
    pooled_std = np.sqrt(((len(x)-1) * np.var(x, ddof=1) 
                          + (len(y)-1) * np.var(y, ddof=1)) / 
                             (len(x) + len(y) - 2))
    return (np.mean(x) - np.mean(y)) / pooled_std
    
def wilcoxon_test(consistencies_matrix, group_A_cells, group_B_cells):
	"""
		Performs an unpaired wilcoxon test (or mann-whitney U test) for each reaction between group_A and group_B
	"""
	#per reaction/meta-reaction, conduct wilcoxon test between group_A and group_B
	group_A = consistencies_matrix.loc[:,group_A_cells]
	group_B = consistencies_matrix.loc[:,group_B_cells]
	results = pd.DataFrame(index = consistencies_matrix.index, columns = ['wilcox_stat', 'wilcox_pval', 'cohens_d'], dtype='float64')
	for rxn in consistencies_matrix.index:
		A, B = group_A.loc[rxn].to_numpy().ravel(), group_B.loc[rxn].to_numpy().ravel()
		#sometimes there's a solitary value, and we don't want to test then
		if len(np.unique(A)) == 1 and len(np.unique(B)) == 1:
			if np.unique(A) == np.unique(B):
				#we've got no data. set p-value to 1 and skip!
				#(p-value needs to be 1 so multipletests doesn't cry)
				results.loc[rxn, ['wilcox_pval']] = 1
				continue
		stat, pval = mannwhitneyu(A, B, alternative='two-sided')
		c_d = cohens_d(A, B)
		results.loc[rxn, ['wilcox_stat', 'wilcox_pval', 'cohens_d']] = stat, pval, c_d
	results['adjusted_pval'] = np.array(multipletests(results['wilcox_pval'], method='fdr_bh')[1], dtype='float64')
	return results

def random_walk(
    g: nx.DiGraph,
    start_node: str,
    node_weights: pd.Series,
    steps: int = 10,
) -> list:
    """
    Performs a random walk on a graph

    Don't allow to return to start node immediately
    """
    current_node = start_node
    banned_node = ''
    path = [current_node]
    for i in range(steps):
        neighbors = list(g.neighbors(current_node))
        if banned_node in neighbors:
            neighbors.remove(banned_node)
            if len(neighbors) == 0:
                continue
        weights = node_weights.loc[neighbors]
        weights = weights.to_numpy().ravel()
        weights = weights / weights.sum()
        banned_node = current_node
        current_node = str(np.random.choice(neighbors, p=weights))
        path.append(current_node)
    return path

def active_custom_pathways(
    g: nx.DiGraph,
    node_weights: pd.Series,
    subwalk_size = 5,
    steps: int = 10,
    walks_per_node: int = 100,
    seed: int = 8
) -> list:
    """
    Determines active pathways based on random walks
    """
    np.random.seed(seed)

    pathways = []
    for node in tqdm(node_weights.index):
        for i in range(walks_per_node):
            pathways.append(random_walk(g, node, node_weights, steps))

    subwalk_counts = defaultdict(int)
    for walk in pathways:
        for sub in get_subwalks(walk, window_size=subwalk_size):
            subwalk_counts[';'.join(sub)] += 1

    subwalks = []
    counts = []
    for sub, count in subwalk_counts.items():
        subwalks.append(sub)
        counts.append(count)

    walk_df = pd.DataFrame({'subwalk': subwalks, 'count': counts})
    walk_df = walk_df.sort_values(by='count', ascending=False)

    return walk_df
 
def pairwise_distance_correlation(embedding1, embedding2, metric='euclidean', size=10000000):
    # Compute pairwise distance matrices
    dist1 = pdist(embedding1, metric=metric)
    dist2 = pdist(embedding2, metric=metric)
    
    # Compute Spearman correlation
    corr, _ = spearmanr(dist1[:size], dist2[:size])
    return corr

def knn_consistency(embedding1, embedding2, k=100, return_mean=True):
    # Find common indices between the two embeddings
    common_indices = embedding1.index.intersection(embedding2.index)
    
    if len(common_indices) == 0:
        raise ValueError("No common indices found between embeddings")
    
    # Subset both embeddings to common indices
    emb1_common = embedding1.loc[common_indices]
    emb2_common = embedding2.loc[common_indices]
    
    # Find k nearest neighbors for each embedding
    nbrs1 = NearestNeighbors(n_neighbors=k).fit(emb1_common) 
    nbrs2 = NearestNeighbors(n_neighbors=k).fit(emb2_common)
    
    knn1 = nbrs1.kneighbors(return_distance=False)
    knn2 = nbrs2.kneighbors(return_distance=False) 
    # Calculate Jaccard index for each cell
    jaccard_indices = []
    for neighbors1, neighbors2 in zip(knn1, knn2):
        set1 = set(neighbors1)
        set2 = set(neighbors2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        jaccard_indices.append(intersection / union if union > 0 else 0)
    
    # Average Jaccard index across all cells
    if return_mean:
        return np.mean(jaccard_indices)
    else:
        return jaccard_indices

def reaction_correlation(rep1, rep2):
    """
    Takes in enzyme activity matrices for two replicates and returns pd.Series of correlations between reactions
    """
    corrs = []
    for col in rep1.columns:
        corrs.append(spearmanr(rep1[col], rep2[col])[0])
    return pd.Series(corrs, rep1.columns)

def calculate_directions(rna, start_cells, end_cells, embedding_key, graph_key=None):
    """
    Calculates directions between start and end cells along with direction loading for each reaction if graph is provided

    Parameters
    ----------
    rna
        anndata object with embedding and graph
    start_cells
        cells to start from
    end_cells
        cells to end at
    embedding_key
        key for embedding
    graph_key
        key for graph to calculate loading of direction

    Returns
    --------
    dir
        direction
    loading (optional)
        loading for each reaction
    """
    if embedding_key not in rna.obsm:
        raise ValueError(f'Embedding key {embedding_key} not found in rna.obsm')

    dir = rna[end_cells,:].obsm[embedding_key].mean(axis=0) - rna[start_cells,:].obsm[embedding_key].mean(axis=0)

    if graph_key is not None:
        if graph_key not in rna.uns:
            raise ValueError(f'Graph key {graph_key} not found in rna.uns')
        loading = pd.Series(np.array(rna.uns[graph_key])@dir, index=rna.uns[graph_key].index)
        return dir, loading
    else:
        return dir
    
def calculate_pathway_scores(enzyme_acts, reaction_info, pathways=None, min_rxns=None):
    """
    Calculates pathway scores for a given enzyme activity matrix and pathways

    Parameters
    ----------
    enzyme_acts
        enzyme activity matrix (cells by reactions)
    reaction_info
        reaction information (rna.uns['Reaction Info'])
    pathways
        pathway ids to calculate scores for (default: all pathways)
    min_rxns
        minimum number of reactions to include a pathway (default: None)

    Returns
    --------
    pathway_acts_df
        pathway activity dataframe (cells by pathways)
    pathway_rxns
        pathway to rxns dictionary

    """
    id_to_name = {}
    name_to_id = {}
    
    for i in range(len(reaction_info)):
        row = reaction_info.iloc[i]
        names = row['Pathway Names'].split(';')
        ids = row['Pathways'].split(';')
        for name, id in zip(names, ids):
            id_to_name[id] = name
            name_to_id[name] = id

    if pathways is None:
        pathways = list(id_to_name.keys())

    # get pathway to rxns
    pathway_rxns = {}
    for id in pathways:
        temp_pathway_rxns = []
        for i in range(len(reaction_info)):
            row = reaction_info.iloc[i]
            ids = row['Pathways'].split(';')
            if id in ids:
                temp_pathway_rxns.append(row.name)
        pathway_rxns[id] = temp_pathway_rxns

    pathways = []
    pathway_acts = []
    for pathway, rxns in pathway_rxns.items():
        if min_rxns is not None and len(rxns) < min_rxns:
            continue
        pathways.append(pathway)
        rxn_acts = enzyme_acts.loc[:,rxns]
        rxn_acts_mean = rxn_acts.mean(axis=1)
        pathway_acts.append(rxn_acts_mean)
    # build (pathways x cells) then transpose to (cells x pathways)
    pathway_acts_df = pd.DataFrame(pathway_acts, index=pathways).dropna().T

    return pathway_acts_df, pathway_rxns, id_to_name


def calculate_ddp_scores(
    enzyme_acts: pd.DataFrame,
    rxn_to_ddp: pd.Series,
) -> tuple[pd.DataFrame, dict]:
    """
    Calculate average reaction score in each cell for each DDP.

    Parameters
    ----------
    enzyme_acts
        Enzyme activity matrix, typically shaped (cells x reactions).
    rxn_to_ddp
        Pandas Series mapping reactions -> DDP. The Series index must be reactions.

    Returns
    -------
    ddp_scores
        DataFrame of DDP scores with shape (cells x ddp), where each entry is the
        mean activity across reactions assigned to that DDP for that cell.
    ddp_rxns
        Dictionary mapping ddp -> list of reactions included.
    """
    if not isinstance(enzyme_acts, pd.DataFrame):
        raise TypeError("enzyme_acts must be a pandas DataFrame")
    if not isinstance(rxn_to_ddp, pd.Series):
        raise TypeError("rxn_to_ddp must be a pandas Series")

    rxn_to_ddp = rxn_to_ddp.dropna()

    overlap_cols = enzyme_acts.columns.intersection(rxn_to_ddp.index)
    overlap_idx = enzyme_acts.index.intersection(rxn_to_ddp.index)

    # If no overlap on columns but overlap on index, assume matrix is (reactions x cells).
    if len(overlap_cols) == 0 and len(overlap_idx) > 0:
        enzyme_acts = enzyme_acts.T
        overlap_cols = enzyme_acts.columns.intersection(rxn_to_ddp.index)

    if len(overlap_cols) == 0:
        raise ValueError(
            "No overlapping reactions found between enzyme_acts and rxn_to_ddp. "
            "Expected rxn_to_ddp.index to match enzyme_acts columns (reactions)."
        )

    rxn_to_ddp = rxn_to_ddp.loc[overlap_cols]

    ddp_rxns: dict = {
        ddp: list(rxn_to_ddp.index[rxn_to_ddp == ddp])
        for ddp in pd.unique(rxn_to_ddp)
    }

    ddps = []
    ddp_scores = []
    for ddp, rxns in ddp_rxns.items():
        if len(rxns) == 0:
            continue
        ddps.append(ddp)
        ddp_scores.append(enzyme_acts.loc[:, rxns].mean(axis=1))

    # build (ddp x cells) then transpose to (cells x ddp)
    ddp_scores_df = pd.DataFrame(ddp_scores, index=ddps).dropna().T
    return ddp_scores_df, ddp_rxns


def calculate_ddps(
    G: nx.Graph,
    corr_df: pd.DataFrame,
    min_corr: float = 0.6,
    min_size: int = 3,
) -> tuple[pd.Series, dict, list[list]]:
    """
    Calculate DDPs by clustering reactions on a graph with a correlation threshold.

    This mirrors the notebook's `get_strict_pathways` + `rxn_clusters` construction:
    - Start with each node in its own cluster.
    - Iteratively merge clusters connected by an edge if the *minimum* correlation
      between any node-pair across the two clusters is >= `min_corr`
      (i.e., complete-linkage on correlation, restricted to graph edges).
    - Keep clusters with size >= `min_size`.

    Parameters
    ----------
    G
        Graph whose nodes are reactions (or meta-reactions). May be a superset of the
        reactions in ``corr_df``; only reactions present in ``corr_df`` are clustered
        and listed in ``rxn_to_ddp``, and only edges between those reactions are used.
    corr_df
        Correlation DataFrame; reactions are ``index`` ∩ ``columns`` (in index order).
        This set may be a subset of ``G.nodes()`` or otherwise differ from ``G``.
    min_corr
        Minimum complete-linkage correlation required to merge two clusters.
    min_size
        Minimum cluster size to keep as a DDP.

    Returns
    -------
    rxn_to_ddp
        Series with one row per reaction in ``corr_df.index`` ∩ ``corr_df.columns``:
        ddp_id (e.g. ``"ddp_0"``) if in a kept DDP, else ``None``. Suitable for
        ``calculate_ddp_scores`` (which ignores unassigned entries via ``dropna``).
    ddp_rxns
        Dict mapping ddp_id (string) -> list of reactions in that DDP.
    clusters
        List of clusters, each a list of reactions.
    """
    in_both = corr_df.index.intersection(corr_df.columns)
    nodes = []
    seen = set()
    for r in corr_df.index:
        if r in in_both and r not in seen:
            seen.add(r)
            nodes.append(r)

    if len(nodes) == 0:
        return pd.Series(dtype=object), {}, []

    # Pre-filter correlation matrix to reactions in corr_df only.
    corr_matrix = corr_df.loc[nodes, nodes].values
    node_idx = {node: i for i, node in enumerate(nodes)}

    # Initialization: each node is its own cluster.
    node_to_cluster = {node: i for i, node in enumerate(nodes)}
    clusters = {i: {node} for i, node in enumerate(nodes)}

    def get_min_corr(c1: set, c2: set) -> float:
        indices1 = [node_idx[n] for n in c1]
        indices2 = [node_idx[n] for n in c2]
        sub_matrix = corr_matrix[np.ix_(indices1, indices2)]
        return float(np.min(sub_matrix))

    # Iterative merging (restricted to physical graph edges).
    while True:
        best_merge = None
        max_min_corr = -1.0

        for u, v in G.edges():
            if u not in node_to_cluster or v not in node_to_cluster:
                continue
            c1_id = node_to_cluster[u]
            c2_id = node_to_cluster[v]
            if c1_id == c2_id:
                continue

            current_min = get_min_corr(clusters[c1_id], clusters[c2_id])
            if current_min >= min_corr and current_min > max_min_corr:
                max_min_corr = current_min
                best_merge = (c1_id, c2_id)

        if best_merge is None:
            break

        id1, id2 = best_merge
        clusters[id1] = clusters[id1].union(clusters[id2])
        for node in clusters[id2]:
            node_to_cluster[node] = id1
        del clusters[id2]

    # Post-process: keep only clusters meeting min_size, then re-index ddp ids.
    kept = [sorted(list(c)) for c in clusters.values() if len(c) >= min_size]
    kept = sorted(kept, key=lambda c: (len(c), c[0] if len(c) else ""), reverse=False)

    ddp_rxns: dict[str, list] = {
        f"ddp_{i}": cluster for i, cluster in enumerate(kept)
    }
    rxn_to_ddp: dict = {n: None for n in nodes}
    for ddp_id, rxns in ddp_rxns.items():
        for rxn in rxns:
            rxn_to_ddp[rxn] = ddp_id

    return pd.Series(rxn_to_ddp, dtype=object), ddp_rxns, kept