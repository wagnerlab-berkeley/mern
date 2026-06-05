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
from scipy.cluster.hierarchy import cophenet
from statsmodels.stats.multitest import multipletests
from collections import defaultdict
from scipy.spatial.distance import pdist, squareform
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
) -> tuple[pd.Series, dict, list[list], np.ndarray]:
    """
    Calculate DDPs by clustering reactions on a graph with a correlation threshold.
    Also returns a SciPy-compatible linkage matrix of the full merge history.

    Returns
    -------
    rxn_to_ddp
        Series mapping reaction -> ddp_id or None.
    ddp_rxns
        Dict mapping ddp_id -> list of reactions.
    clusters
        List of kept clusters.
    linkage_matrix
        A NumPy array of shape ``(len(nodes) - 1, 4)`` tracking the hierarchical
        merge history. Distance is represented as ``1.0 - correlation`` for
        graph-supported merges. Disconnected graph roots are merged at an
        artificial high distance so SciPy can render a complete dendrogram.
    """
    in_both = corr_df.index.intersection(corr_df.columns)
    nodes = []
    seen = set()
    for r in corr_df.index:
        if r in in_both and r not in seen:
            seen.add(r)
            nodes.append(r)

    if len(nodes) == 0:
        return pd.Series(dtype=object), {}, [], np.empty((0, 4))

    # Pre-filter correlation matrix to reactions in corr_df only.
    corr_matrix = corr_df.loc[nodes, nodes].values
    node_idx = {node: i for i, node in enumerate(nodes)}

    # Initialization: each node is its own cluster.
    node_to_cluster = {node: i for i, node in enumerate(nodes)}
    clusters = {i: {node} for i, node in enumerate(nodes)}

    linkage_rows = []
    cluster_to_linkage_id = {i: i for i in range(len(nodes))}
    next_linkage_id = len(nodes)

    def get_min_corr(c1: set, c2: set) -> float:
        indices1 = [node_idx[n] for n in c1]
        indices2 = [node_idx[n] for n in c2]
        sub_matrix = corr_matrix[np.ix_(indices1, indices2)]
        return float(np.min(sub_matrix))

    ddp_clusters = None

    # Iterative complete-linkage merging, restricted to physical graph edges.
    # DDP calls are frozen at min_corr, but linkage continues below that cutoff.
    while True:
        best_merge = None
        max_min_corr = -np.inf

        for u, v in G.edges():
            if u not in node_to_cluster or v not in node_to_cluster:
                continue
            c1_id = node_to_cluster[u]
            c2_id = node_to_cluster[v]
            if c1_id == c2_id:
                continue

            current_min = get_min_corr(clusters[c1_id], clusters[c2_id])
            if current_min > max_min_corr:
                max_min_corr = current_min
                best_merge = (c1_id, c2_id)

        if best_merge is None:
            break

        id1, id2 = best_merge
        if max_min_corr < min_corr and ddp_clusters is None:
            ddp_clusters = {k: set(v) for k, v in clusters.items()}

        linkage_rows.append([
            float(cluster_to_linkage_id[id1]),
            float(cluster_to_linkage_id[id2]),
            float(1.0 - max_min_corr),
            float(len(clusters[id1]) + len(clusters[id2])),
        ])

        cluster_to_linkage_id[id1] = next_linkage_id
        next_linkage_id += 1

        clusters[id1] = clusters[id1].union(clusters[id2])
        for node in clusters[id2]:
            node_to_cluster[node] = id1
        del clusters[id2]
        del cluster_to_linkage_id[id2]

    if ddp_clusters is None:
        ddp_clusters = {k: set(v) for k, v in clusters.items()}

    remaining_roots = [(cluster_to_linkage_id[k], len(v)) for k, v in clusters.items()]
    last_dist = linkage_rows[-1][2] if linkage_rows else 0.0
    artificial_dist = max(2.0, last_dist + 0.1)

    while len(remaining_roots) > 1:
        linkage_id1, size1 = remaining_roots.pop()
        linkage_id2, size2 = remaining_roots.pop()

        linkage_rows.append([
            float(linkage_id1),
            float(linkage_id2),
            float(artificial_dist),
            float(size1 + size2),
        ])

        remaining_roots.append((next_linkage_id, size1 + size2))
        next_linkage_id += 1

    # Post-process: keep only clusters meeting min_size, then re-index ddp ids.
    kept = [sorted(list(c)) for c in ddp_clusters.values() if len(c) >= min_size]
    kept = sorted(kept, key=lambda c: (len(c), c[0] if len(c) else ""), reverse=False)

    ddp_rxns: dict[str, list] = {
        f"ddp_{i}": cluster for i, cluster in enumerate(kept)
    }
    rxn_to_ddp: dict = {n: None for n in nodes}
    for ddp_id, rxns in ddp_rxns.items():
        for rxn in rxns:
            rxn_to_ddp[rxn] = ddp_id

    linkage_matrix = np.array(linkage_rows) if linkage_rows else np.empty((0, 4))

    return pd.Series(rxn_to_ddp, dtype=object), ddp_rxns, kept, linkage_matrix


def calculate_cophenetic_corr_matrix(
    linkage_matrix: np.ndarray,
    labels: list,
) -> pd.DataFrame:
    """
    Convert a DDP linkage matrix into pairwise cophenetic correlations.

    The linkage matrix from ``calculate_ddps`` stores distance as
    ``1.0 - correlation``. This helper converts SciPy's cophenetic distances
    back to correlations and returns a square reaction-by-reaction matrix.
    """
    labels = list(labels)
    n_labels = len(labels)

    if n_labels == 0:
        return pd.DataFrame(index=labels, columns=labels, dtype=float)
    if n_labels == 1:
        return pd.DataFrame([[1.0]], index=labels, columns=labels)

    expected_rows = n_labels - 1
    if linkage_matrix.shape != (expected_rows, 4):
        raise ValueError(
            "linkage_matrix must have shape "
            f"({expected_rows}, 4) for {n_labels} labels."
        )

    cophenetic_distances = cophenet(linkage_matrix)
    cophenetic_corr = 1.0 - squareform(cophenetic_distances)
    np.fill_diagonal(cophenetic_corr, 1.0)

    return pd.DataFrame(cophenetic_corr, index=labels, columns=labels)


def compare_cophenetic_corr(
    wt_linkage_matrix: np.ndarray,
    ko_linkage_matrix: np.ndarray,
    labels: list,
    graph: nx.Graph | None = None,
) -> pd.DataFrame:
    """
    Compare WT and KO cophenetic correlations as ``WT - KO``.

    If ``graph`` is provided, only direct graph edges among ``labels`` are
    returned. Otherwise, all unordered reaction pairs are returned.
    """
    labels = list(labels)
    label_set = set(labels)
    wt_cophenetic = calculate_cophenetic_corr_matrix(wt_linkage_matrix, labels)
    ko_cophenetic = calculate_cophenetic_corr_matrix(ko_linkage_matrix, labels)

    if graph is None:
        pairs = itertools.combinations(labels, 2)
    else:
        seen_pairs = set()
        graph_pairs = []
        for u, v in graph.edges():
            if u not in label_set or v not in label_set or u == v:
                continue
            pair_key = frozenset((u, v))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            graph_pairs.append((u, v))
        pairs = graph_pairs

    rows = []
    for rxn_1, rxn_2 in pairs:
        wt_corr = wt_cophenetic.loc[rxn_1, rxn_2]
        ko_corr = ko_cophenetic.loc[rxn_1, rxn_2]
        rows.append({
            "rxn_1": rxn_1,
            "rxn_2": rxn_2,
            "wt_cophenetic_corr": wt_corr,
            "ko_cophenetic_corr": ko_corr,
            "delta_wt_minus_ko": wt_corr - ko_corr,
        })

    return pd.DataFrame(
        rows,
        columns=[
            "rxn_1",
            "rxn_2",
            "wt_cophenetic_corr",
            "ko_cophenetic_corr",
            "delta_wt_minus_ko",
        ],
    )
