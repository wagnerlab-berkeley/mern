from typing import Callable, List, Optional, Union, Mapping, Tuple
import matplotlib.axes as ma
import numpy as np
import pandas as pd
import seaborn as sns
import scanpy as sc
from matplotlib import rcParams
import matplotlib.pyplot as plt
import networkx as nx
from anndata import AnnData
import matplotlib.cm as cm
from networkx.drawing.nx_agraph import graphviz_layout as _nx_graphviz_layout
import plotly.graph_objects as go
import plotly.express as px

from .add_edge import addEdge
from .metabolic_datasets import MetabolicDataset


def _graph_layout(graph: Union[nx.Graph, nx.DiGraph]) -> Mapping:
    """
    Return graph coordinates, preferring Graphviz and falling back when
    pygraphviz is not installed.
    """
    if len(graph.nodes) == 0:
        return {}

    try:
        return _nx_graphviz_layout(graph)
    except ImportError:
        # Keep a stable fallback layout so plots remain usable without
        # system-level Graphviz headers or pygraphviz.
        return nx.spring_layout(graph, seed=42)


def training_plot(
    train_metrics: Mapping,
    val_metrics: Mapping,
    ylim: float = 1.0
) -> ma.Axes:
    """
    Plot training metrics of VAE
    
    Parameters
    ----------
    train_metrics
        dictionary with key indicating metric and values as list for training
    val_metrics
        same as train_metrics, but contains metrics on validation set
    ylim
        y limit of plot
    Returns
    --------
    ax
        Plot axes
    """
    # Generate a color map
    num_metrics = len(train_metrics)
    colors = plt.cm.tab10(np.linspace(0, 1, num_metrics))

    # Create a plot
    fig, ax = plt.subplots(figsize=(10, 6))
    #fig, ax = plt.subplots()

    # Plot training metrics
    for idx, (metric, values) in enumerate(train_metrics.items()):
        ax.plot(values, label=f'Train {metric}', color=colors[idx], alpha=1.0, linestyle='-')

    # Plot validation metrics
    val_x = range(0, len(values) + 1, 10)
    for idx, (metric, values) in enumerate(val_metrics.items()):
        ax.plot(val_x, values, label=f'Val {metric}', color=colors[idx], alpha=0.5, linestyle='--')

    # Add titles and labels
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_ylim(0, ylim)
    ax.legend(bbox_to_anchor=(1, 1), loc='upper left', handlelength=3)

    # Adjust layout to make room for the legend
    plt.tight_layout()

    return ax
    
def metabolic_topology_plot(
    graph: nx.DiGraph,
    rna: AnnData,
    rxn_info_key: str,
    display_modules: List = [],
    display_rxn_modules: List = [],
    fig_size: Tuple = (25, 25),
    class_type: str = 'Broad',
    node_size: int = 300,
    with_labels: bool = True
) -> ma.Axes:
    """
    General function for displaying metabolic topology graphs

    Parameters
    ----------
    graph
        metabolic topology graph
    rna
        anndata object
    rxn_info_key
        key for rxn info in anndata
    species
        species for which to get module information
    display_modules
        list of modules to highlight
    display_rxn_modules
        list of rxns to highlight modules for
    fig_size
        size of figure
    class_type
        whether to use broad or narrow class for module description
    node_size
    with_labels    
    Returns
    --------
    fig
        met. topology figure
    """

    assert rxn_info_key in rna.uns, "Add module info first."

    reaction_info = rna.uns[rxn_info_key]

    fig = plt.figure(figsize=fig_size)

    # Add an axes to the figure
    ax = fig.add_subplot(111)

    node_labels = []
    if len(display_modules) > 0 or len(display_rxn_modules) > 0:
        
        for node in graph.nodes:
            node_modules = []

            # check if rxn is in display modules
            node_info = reaction_info.loc[node]
            
            temp_mods = node_info['Modules'].split(';')
            temp_mods_names = node_info['Names'].split(';')
            for i, mod in enumerate(temp_mods):
                if mod in display_modules:
                    node_modules.append(temp_mods_names[i])

            if node in display_rxn_modules:
                node_modules += temp_mods_names
            
            node_modules = list(set(node_modules))
            node_modules = ';'.join(node_modules)
            node_labels.append(node_modules)

    else:
        for node in graph.nodes:
            node_info = reaction_info.loc[node]
            node_info = node_info[f'{class_type} Classes']
            node_info = list(set(node_info.split(';')))
            node_info = sorted(node_info)
            node_labels.append(';'.join(node_info))

    unique_labels = list(set(node_labels))

    # Generate a colormap
    cmap = cm.get_cmap('tab20', len(unique_labels))  # You can change 'viridis' to any other colormap
    color_dict = {label: cmap(i) for i, label in enumerate(unique_labels)}

    node_colors = [color_dict[label] for label in node_labels]

    # Draw the graph
    nx.draw(graph, _graph_layout(graph), with_labels=with_labels, node_color=node_colors, cmap=cmap, ax=ax, node_size=node_size)

    # Create a legend
    for label in color_dict.keys():
        plt.plot([], [], 'o', label=label, color=color_dict[label])

    ax.legend(loc='best', title="Reaction Labels")

    return ax

def metabolic_topology_plot_go(
    graph: nx.DiGraph,
    rna: AnnData,
    rxn_info_key: str,
    display_modules: List = [],
    display_rxn_modules: List = [],
    class_type: str = 'Broad',
    embedding_dim_name: str = 'Rxn Embeddings',
    display_embedding_dim: int = None,
    subset_graph: bool = False,
    user_labels: pd.Series = None,
    hop_rxn: str = None,
    hops: int = 1,
) -> ma.Axes:
    """
    General function for displaying metabolic topology graphs

    Parameters
    ----------
    graph
        metabolic topology graph
    rna
        anndata object
    rxn_info_key
        key for rxn info in anndata
    display_modules
        list of modules to highlight
    display_rxn_modules
        list of rxns to highlight modules for
    fig_size
        size of figure
    class_type
        whether to use broad or narrow class for module description
    embedding_dim_name
        where rxn embeddings are stored in anndata.uns
    display_embedding_dim
        which embedding dimension to splay
    subset_graph
        whether to only display modules
    user_labels
        pandas series with rxns as index indicating continuous value to plot on graph
    hop_rxn
        rxn to construct graph around
    hops
        how many hops to go from rxn
    Returns
    --------
    fig
        met. topology figure
    """

    assert rxn_info_key in rna.uns, "Add module info first."

    reaction_info = rna.uns[rxn_info_key]

    continuous_label = False
    node_labels = []
    node_text = []
    include_node = []
    if len(display_modules) > 0 or len(display_rxn_modules) > 0:
        
        for node in graph.nodes:
            node_modules = []

            # check if rxn is in display modules
            node_info = reaction_info.loc[node]

            temp_mods = node_info['Modules'].split(';')
            temp_mods_names = node_info['Names'].split(';')
            for i, mod in enumerate(temp_mods):
                if mod in display_modules:
                    node_modules.append(temp_mods_names[i])

            if node in display_rxn_modules:
                node_modules += temp_mods_names

            if len(node_modules) > 0:
                include_node.append(True)
            else: include_node.append(False)
            node_modules = list(set(node_modules))
            node_modules = ';'.join(node_modules)

            graph.nodes[node]['label'] = node_modules
            #node_labels.append(node_modules)
            rxn_name = node_info['Rxn Name']
            graph.nodes[node]['text'] = f'{node}: {rxn_name}'
            #node_text.append(f'{node}: {rxn_name}')
            
    if display_embedding_dim is not None:
        assert embedding_dim_name in rna.uns, "Add rxn embeddings to anndata object."
        
        continuous_label = True
    
        for node in graph.nodes:
            node_info = reaction_info.loc[node]
            graph.nodes[node]['label'] = rna.uns[embedding_dim_name].loc[node, display_embedding_dim]
            #node_labels.append(rna.uns[embedding_dim_name].loc[node, display_embedding_dim])
            rxn_name = node_info['Rxn Name']
            rxn_modules = node_info['Modules']
            rxn_module_names = node_info['Names']
            graph.nodes[node]['text'] = f'{node}: {rxn_name}<br>{rxn_modules}: {rxn_module_names}'
            #node_text.append(f'{node}: {rxn_name}<br>{rxn_modules}: {rxn_module_names}')

    def get_hop_neighbors(g, node, hops):
        if hops == 0: return [node]
            
        to_return = [node]
        neighbors = list(g.neighbors(node))
        for n in neighbors:
            more_n = get_hop_neighbors(g, n, hops-1)
            to_return += get_hop_neighbors(g, n, hops-1)
            
        return to_return
        
    if hop_rxn is not None:

        nodes_to_include = get_hop_neighbors(graph, hop_rxn, hops)

        for node in graph.nodes:
            if node in nodes_to_include: include_node.append(True)
            else: include_node.append(False)

    if user_labels is not None:
        assert sorted(list(user_labels.index)) == sorted(list(graph.nodes)), "Label index does not match graph nodes"
        
        continuous_label = True

        for node in graph.nodes:
            node_info = reaction_info.loc[node]
            graph.nodes[node]['label'] = user_labels[node]
            #node_labels.append(user_labels[node])
            rxn_name = node_info['Rxn Name']
            rxn_modules = node_info['Modules']
            rxn_module_names = node_info['Names']
            graph.nodes[node]['text'] = f'{node}: {rxn_name}<br>{rxn_modules}: {rxn_module_names}'
            #node_text.append(f'{node}: {rxn_name}<br>{rxn_modules}: {rxn_module_names}')
        
    if user_labels is None and display_embedding_dim is None and len(display_modules) == 0 and len(display_rxn_modules)==0:
        for node in graph.nodes:
            node_info = reaction_info.loc[node]
            rxn_name = node_info['Rxn Name']
            node_info = node_info[f'{class_type} Classes']
            node_info = list(set(node_info.split(';')))
            node_info = sorted(node_info)
            graph.nodes[node]['label'] = ';'.join(node_info)
            #node_labels.append(';'.join(node_info))
            graph.nodes[node]['text'] = f'{node}: {rxn_name}'
            #node_text.append(f'{node}: {rxn_name}')

    # subset to only display modules + immediate neighbors
    if subset_graph:
        subset_nodes = np.array(graph.nodes)[include_node]
        graph = graph.subgraph(subset_nodes)
        pos = _graph_layout(graph)
    else:
        pos = _graph_layout(graph)

    node_text = list(nx.get_node_attributes(graph, 'text').values())
    node_labels = list(nx.get_node_attributes(graph, 'label').values())
    nodeSize = 10
    lineWidth = 2
    lineColor = '#888'
    edge_x = []
    edge_y = []

    for edge in graph.edges():
        start = pos[edge[0]]
        end = pos[edge[1]]
        edge_x, edge_y = addEdge(start, end, edge_x, edge_y, 0.9, 'end', 10, 20, nodeSize)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=lineWidth, color=lineColor),
        hoverinfo='none',
        mode='lines',
        showlegend=False  # Do not show legend for edge trace
    )

    node_x = []
    node_y = []

    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    if continuous_label:
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                #showscale=True,
                # colorscale options
                #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
                #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
                #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
                colorscale='YlGnBu',
                reversescale=True,
                color=node_labels,
                size=nodeSize,
                colorbar=dict(
                    thickness=15,
                    title='Rxn Loading',
                    xanchor='left',
                ),
                line_width=2,
                opacity=1.0),
            showlegend=True  # Do not show legend for node trace
        )
        
    else:
        # Create a mapping of unique labels to colors
        unique_labels = list(set(node_labels))
        color_map = px.colors.qualitative.Dark24
        color_dict = {label: color_map[i % len(color_map)] for i, label in enumerate(unique_labels)}
        node_colors = [color_dict[label] for label in node_labels]
    
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                color=node_colors,
                size=nodeSize,
                line_width=2,
                opacity=1.0,
            ),
            showlegend=False  # Do not show legend for node trace
        )

    node_trace.text = node_text

    # Create the figure
    fig = go.Figure()

    # Add edge trace
    fig.add_trace(edge_trace)

    # Add the node trace
    fig.add_trace(node_trace)

    if not continuous_label:
        # Add scatter traces for each label to create a legend
        for label in unique_labels:
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(
                    size=10,
                    color=color_dict[label],
                ),
                legendgroup=label,
                showlegend=True,
                name=label
            ))

    # Update layout to include a legend
    fig.update_layout(
        title='<br>Metabolic Topology',
        showlegend=False if continuous_label else True,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=1000, 
        legend=dict(
            title='Node Labels',
            itemsizing='constant'
        )
    )



    """marker=dict(
            #showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            #colorscale='YlGnBu',
            #reversescale=True,
            color=[],
            size=nodeSize,
            #colorbar=dict(
            #    thickness=15,
            #    title='Node Connections',
            #    xanchor='left',
            #    titleside='right'
            #),
            line_width=2))
    
    #node_trace.marker.color = node_colors
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='<br>Metabolic Topology',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    """
    return fig

def infer_coords(node, graph):
    neighbors = list(graph.neighbors(node))
    # Get positions from neighbors that have both 'x' and 'y' attributes
    known_positions = [(n, (graph.nodes[n]['x'], graph.nodes[n]['y']))
                       for n in neighbors if 'x' in graph.nodes[n] and 'y' in graph.nodes[n]]
    if not known_positions:
        return (0, 0)
    # If there are multiple neighbors, return the average position.
    if len(known_positions) > 1:
        avg_x = sum(pos[0] for _, pos in known_positions) / len(known_positions)
        avg_y = sum(pos[1] for _, pos in known_positions) / len(known_positions)
        return (avg_x, avg_y)
    else:
        # Exactly one neighbor with a known position.
        neighbor, pos = known_positions[0]
        # Fetch neighbor's other neighbors (exclude the current node)
        neighbor_neighbors = list(graph.neighbors(neighbor))
        if node in neighbor_neighbors:
            neighbor_neighbors.remove(node)
        # Filter those neighbors with known positions
        neighbor_known = [(n, (graph.nodes[n]['x'], graph.nodes[n]['y']))
                          for n in neighbor_neighbors if 'x' in graph.nodes[n] and 'y' in graph.nodes[n]]
        offset = 100  # Arbitrary offset distance
        if neighbor_known:
            # Compute the centroid of the neighbor's other known neighbors
            avg_x_nn = sum(pos[0] for _, pos in neighbor_known) / len(neighbor_known)
            avg_y_nn = sum(pos[1] for _, pos in neighbor_known) / len(neighbor_known)
            # Compute the vector from the centroid to the neighbor
            vector_x = pos[0] - avg_x_nn
            vector_y = pos[1] - avg_y_nn
            norm = (vector_x ** 2 + vector_y ** 2) ** 0.5
            if norm == 0:
                # If the centroid coincides with the neighbor, fallback to a default horizontal offset.
                return (pos[0] + offset, pos[1])
            else:
                # Place the node in the direction away from the neighbor's neighbors.
                return (pos[0] + offset * (vector_x / norm), pos[1] + offset * (vector_y / norm))
        # Fallback if the neighbor has no other neighbor with a known position.
        return (pos[0] + offset, pos[1])

def kegg_pathway_plot(
        pathway: str,
        dataset: MetabolicDataset,
        rna: AnnData,
        user_labels: pd.Series = None,
        edge_sizes: pd.Series = None,
        color_scale: str = 'RdBU_r',
        node_labels: pd.Series = None,
        node_color_scale: str = 'RdBU_r',
        node_sizes: pd.Series = None,
        default_edge_size: int = 5,
        show_node_labels: bool = True,
        edge_color_limits: Tuple[float, float] = None,
        node_color_limits: Tuple[float, float] = None,
):
    """
    Create a pathway visualization for a specific KEGG pathway.
    
    Parameters
    ----------
    pathway : str
        KEGG pathway identifier
    dataset : MetabolicDataset
        Dataset containing reaction information
    rna : AnnData
        RNA expression data
    user_labels : pd.Series, optional
        Labels for coloring edges
    edge_sizes : pd.Series, optional
        Sizes for edges
    color_scale : str, optional
        Color scale for edge colors (default: 'RdBU_r')
    node_labels : pd.Series, optional
        Labels for coloring nodes (continuous values indexed by KEGG compound ID)
    node_color_scale : str, optional
        Color scale for node colors (default: 'RdBU_r')
    node_sizes : pd.Series, optional
        Sizes for nodes (continuous values indexed by KEGG compound ID)
    default_edge_size : int, optional
        Default size for edges (default: 5)
    show_node_labels : bool, optional
        Whether to show node labels (default: True)
    edge_color_limits : Tuple[float, float], optional
        Tuple of (min, max) values for edge color scale limits. If None, limits are auto-calculated.
    node_color_limits : Tuple[float, float], optional
        Tuple of (min, max) values for node color scale limits. If None, limits are auto-calculated.
        
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    compound_graph = nx.Graph()
    use_rxns = []
    for reaction in dataset.rxns:
        use_rxn = False
        for r in reaction.name.split(' '):
            r_pathways = [p[0] for p in dataset.rxn_info[r.split('rn:')[1]]['PATHWAY']]
            if pathway in r_pathways:
                use_rxn = True
                use_rxns.append(reaction.name)
                break
        if not use_rxn: continue
        substrates = [s.name for s in reaction.substrates]
        products = [p.name for p in reaction.products]
        
        # Add edges for each substrate-product pair
        for sub in substrates:
            for prod in products:
                if user_labels is not None:
                    value = user_labels.get(reaction.name, 0)
                else:
                    value = 0
                compound_graph.add_edge(
                    sub, prod,
                    reaction=reaction.name,
                    value=value,
                    size=edge_sizes.get(reaction.name, default_edge_size) if edge_sizes is not None else default_edge_size
                )
                
        # Store compound positions from KGML graphics
        for compound in substrates + products:
            try:
                kgml_entry = dataset._get_kgml_entry(compound, pathway)  # Implement KGML entry lookup
            except ValueError:
                continue
            if kgml_entry and kgml_entry.graphics:
                compound_graph.nodes[compound].update({
                    'x': kgml_entry.graphics[0].x,
                    'y': kgml_entry.graphics[0].y
                })
        for node in compound_graph.nodes():
            if 'x' not in compound_graph.nodes[node] or 'y' not in compound_graph.nodes[node]:
                print(node)
                x, y = infer_coords(node, compound_graph)
                compound_graph.nodes[node].update({'x': x, 'y': y}) 

    """Visualize compound net  work with reaction-colored edges"""
    # Node positions from KGML data
    pos = {node: (data['x'], data['y']) for node, data in compound_graph.nodes(data=True)}
    
    # Scale edge sizes to be between 0.5 and 10
    if edge_sizes is not None:
        subset_use_rxns = [r for r in use_rxns if r in edge_sizes.index]
        if subset_use_rxns:
            min_edge = edge_sizes[subset_use_rxns].min()
            max_edge = edge_sizes[subset_use_rxns].max()
            if max_edge - min_edge != 0:
                edge_sizes_scaled = (edge_sizes - min_edge) / (max_edge - min_edge) * (10 - 2) + 2
            else:
                edge_sizes_scaled = pd.Series(2, index=edge_sizes.index)
            for u, v, data in compound_graph.edges(data=True):
                r = data['reaction']
                compound_graph.edges[u, v]['size'] = edge_sizes_scaled.get(r, 2)
        else:
            for u, v in compound_graph.edges():
                compound_graph.edges[u, v]['size'] = 2
    else:
        for u, v in compound_graph.edges():
            compound_graph.edges[u, v]['size'] = 2

    # Calculate normalized values from your pd.Series
    if user_labels is not None:
        subset_use_rxns = []
        # some use rxns are not in the graph
        for r in use_rxns:
            if r in user_labels.index: subset_use_rxns.append(r)
        if subset_use_rxns:
            values = user_labels[subset_use_rxns]
            if edge_color_limits is not None:
                min_val, max_val = edge_color_limits
            else:
                max_abs = max(values.abs().max(), 1e-6)  # Ensure we don't divide by zero
                min_val = -max_abs
                max_val = max_abs
        else:
            if edge_color_limits is not None:
                min_val, max_val = edge_color_limits
            else:
                min_val = -1
                max_val = 1
    else:
        if edge_color_limits is not None:
            min_val, max_val = edge_color_limits
        else:
            min_val = -1
            max_val = 1
    
    # Create edge traces with midpoints for hover
    edge_traces = []
    mid_x, mid_y, edge_text = [], [], []
    
    for u, v, data in compound_graph.edges(data=True):
        norm_value = (data['value'] - min_val) / (max_val - min_val) if max_val != min_val else 0.5
        color = px.colors.sample_colorscale(color_scale, [norm_value])[0]
        
        # Edge line trace
        edge_traces.append(go.Scatter(
            x=[pos[u][0], pos[v][0], None],
            y=[pos[u][1], pos[v][1], None],
            line=dict(width=data['size'], color=color),
            mode='lines',
            hoverinfo='none',  # Disable line hover
            showlegend=False
        ))
        
        # Calculate midpoint for hover marker
        mid_x.append((pos[u][0] + pos[v][0])/2)
        mid_y.append((pos[u][1] + pos[v][1])/2)
        edge_text.append(f"Reaction: {data['reaction']}<br>Value: {data['value']:.2f}<br>Size: {data['size']:.2f}")

    # Create invisible hover markers trace
    hover_trace = go.Scatter(
        x=mid_x,
        y=mid_y,
        mode='markers',
        marker=dict(size=10, opacity=0),  # Invisible markers
        hoverinfo='text',
        text=edge_text,
        showlegend=False
    )

    compound_labels = []
    node_colors = []
    hover_texts = []
    
    # Calculate node color normalization if node_labels provided
    node_min = None
    node_max = None
    if node_labels is not None:
        # Get node values for compounds that exist in the graph
        node_values = []
        for node in compound_graph.nodes():
            if node in node_labels.index:
                node_values.append(node_labels[node])
        
        if node_values:
            if node_color_limits is not None:
                node_min, node_max = node_color_limits
            else:
                max_abs_node = max(abs(max(node_values)), abs(min(node_values)), 1e-6)
                node_min = -max_abs_node
                node_max = max_abs_node
    
    for compound in compound_graph.nodes():
        if 'cpd:' in compound:
            s = compound.split('cpd:')
        elif 'gl:' in compound:
            s = compound.split('gl:')

        try:
            name = dataset.compound_info[s[1]]['name']
            compound_labels.append(name)
            hover_texts.append(f"Name: {name}<br>ID: {compound}")
        except KeyError:
            compound_labels.append(s[1])
            hover_texts.append(f"ID: {compound}")
            
        # Set node color based on normalized node_labels if provided
        if node_labels is not None and compound in node_labels.index and node_min is not None and node_max is not None:
            norm_value = (node_labels[compound] - node_min) / (node_max - node_min)
            node_colors.append(px.colors.sample_colorscale(node_color_scale, [norm_value])[0])
            hover_texts[-1] += f"<br>Value: {node_labels[compound]:.2f}"
        else:
            node_colors.append('lightgray')
            
        # Add size information to hover text if node_sizes provided
        if node_sizes is not None and compound in node_sizes.index:
            hover_texts[-1] += f"<br>Size: {node_sizes[compound]:.2f}"

    # Calculate node sizes if node_sizes provided
    node_size_values = []
    if node_sizes is not None:
        # Get node size values for compounds that exist in the graph
        size_values = []
        for node in compound_graph.nodes():
            if node in node_sizes.index:
                size_values.append(node_sizes[node])
        
        if size_values:
            min_size = min(size_values)
            max_size = max(size_values)
            # Scale sizes to range [12, 20] keeping minimum at 12 (original default)
            if max_size - min_size != 0:
                for node in compound_graph.nodes():
                    if node in node_sizes.index:
                        scaled_size = 12 + (node_sizes[node] - min_size) / (max_size - min_size) * (20 - 12)
                        node_size_values.append(scaled_size)
                    else:
                        node_size_values.append(12)  # Default size for missing values
            else:
                node_size_values = [12] * len(compound_graph.nodes())
        else:
            node_size_values = [12] * len(compound_graph.nodes())
    else:
        node_size_values = [12] * len(compound_graph.nodes())

    # Node trace
    node_x = [pos[n][0] for n in compound_graph.nodes()]
    node_y = [pos[n][1] for n in compound_graph.nodes()]
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text' if show_node_labels else 'markers',
        text=compound_labels if show_node_labels else None,
        hovertext=hover_texts,
        marker=dict(size=node_size_values, color=node_colors, opacity=1.0),
        hoverinfo='text',
        textposition='top center',
        hoverlabel=dict(bgcolor='white'),
        hovertemplate='%{hovertext}<extra></extra>'
    )
    
    # Create colorbar traces for both edges and nodes
    edge_colorbar = go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(
            colorscale=color_scale,
            showscale=True,
            cmin=min_val,
            cmax=max_val,
            colorbar=dict(
                title='Edge Value',
                thickness=15,
                len=0.5,
                yanchor='middle',
                y=0.5,
                xanchor='left',
                x=1.05
            )
        ),
        hoverinfo='none',
        showlegend=False
    )
    
    if node_labels is not None and node_min is not None and node_max is not None:
        node_colorbar = go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(
                colorscale=node_color_scale,
                showscale=True,
                cmin=node_min,
                cmax=node_max,
                colorbar=dict(
                    title='Node Value',
                    thickness=15,
                    len=0.5,
                    yanchor='middle',
                    y=0.5,
                    xanchor='left',
                    x=1.15
                )
            ),
            hoverinfo='none',
            showlegend=False
        )
        colorbar_traces = [edge_colorbar, node_colorbar]
    else:
        colorbar_traces = [edge_colorbar]
    
    # Assemble figure with correct layer order
    fig = go.Figure(data=edge_traces + [node_trace, hover_trace] + colorbar_traces)
    fig.update_layout(
        title='Metabolic Pathway Visualization',
        plot_bgcolor='white',
        hovermode='closest',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False, autorange='reversed'),
        height=800
    )
    return fig

def plot_differential_scores(data, title, c='black'):
    plt.figure(figsize=(10,10))
    axs = plt.gca()
    axs.scatter(data['cohens_d'], -np.log10(data['adjusted_pval']), c=c)
    axs.set_xlabel("Cohen's d", fontsize=16)
    axs.set_ylabel("-log10 (Wilcoxon-adjusted p)", fontsize=16)
    #Everything after this should be tweaked depending on your application
    axs.set_xlim(-2.2, 2.2)
    axs.axvline(0, dashes=(3,3), c='black')
    axs.axhline(1, dashes=(3,3), c='black')
    axs.set_title(title, fontdict={'fontsize':20})
    #axs.annotate('', xy=(0.5, -0.08), xycoords='axes fraction', xytext=(0, -0.08), 
    #       arrowprops=dict(arrowstyle="<-", color='#348C73', linewidth=4))
    #axs.annotate('Th17p', xy=(0.75, -0.12), xycoords='axes fraction', fontsize=16)
    #axs.annotate('', xy=(0.5, -0.08), xycoords='axes fraction', xytext=(1, -0.08), 
    #        arrowprops=dict(arrowstyle="<-", color='#E92E87', linewidth=4))
    #axs.annotate('Th17n', xy=(0.25, -0.12), xycoords='axes fraction', fontsize=16)
    """for r in data.index:
        if r in labeled_reactions:
            x = data.loc[r, 'cohens_d']
            y = -np.log10(data.loc[r, 'adjusted_pval'])
            offset = (20, 0)
            if x < 0:
                offset = (-100, -40)
            axs.annotate(labeled_reactions[r], (x,y), xytext = offset, 
                         textcoords='offset pixels', arrowprops={'arrowstyle':"-"})"""

def pick_nodes_from_top_edges(
    g: nx.Graph,
    top_k_edges: int = 25,
    value_attr: str = "value",
    abs_value: bool = True
) -> set:
    edges = []
    for u, v, data in g.edges(data=True):
        val = data.get(value_attr, 0.0)
        score = abs(val) if abs_value else val
        edges.append((score, u, v))
    edges.sort(reverse=True, key=lambda x: x[0])
    chosen = edges[:top_k_edges]
    label_nodes = set()
    for _, u, v in chosen:
        label_nodes.add(u); label_nodes.add(v)
    return label_nodes
    
def custom_pathway_plot(
        rxns: List[str],
        dataset: MetabolicDataset,
        rna: AnnData,
        user_labels: pd.Series = None,
        edge_sizes: pd.Series = None,
        color_scale: str = 'RdBU_r',
        node_labels: pd.Series = None,
        node_color_scale: str = 'RdBU_r',
        node_sizes: pd.Series = None,
        edge_color_limits: Tuple[float, float] = None,
        node_color_limits: Tuple[float, float] = None,
        show_node_labels: bool = False,
        top_k_edges: int = 25,
):
    """
    Create a pathway visualization using graphviz layout for a custom list of reactions.
    
    Parameters
    ----------
    rxns : List[str]
        List of reaction IDs to include in the pathway
    dataset : KeggMetabolicDataset
        Dataset containing reaction information
    rna : AnnData
        RNA expression data
    user_labels : pd.Series, optional
        Labels for coloring edges
    edge_sizes : pd.Series, optional
        Sizes for edges
    color_scale : str, optional
        Color scale for edge colors (default: 'RdBU_r')
    node_labels : pd.Series, optional
        Labels for coloring nodes (continuous values indexed by KEGG compound ID)
    node_color_scale : str, optional
        Color scale for node colors (default: 'RdBU_r')
    node_sizes : pd.Series, optional
        Sizes for nodes (continuous values indexed by KEGG compound ID)
    edge_color_limits : Tuple[float, float], optional
        Tuple of (min, max) values for edge color scale limits. If None, limits are auto-calculated.
    node_color_limits : Tuple[float, float], optional
        Tuple of (min, max) values for node color scale limits. If None, limits are auto-calculated.
        
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    compound_graph = nx.Graph()
    use_rxns = []
    
    # Build the graph structure
    for reaction in dataset.rxns:
        # Check if this reaction is in our custom list
        if reaction.name in rxns:
            use_rxns.append(reaction.name)
            
            substrates = [s.name for s in reaction.substrates]
            products = [p.name for p in reaction.products]
            
            # Add edges for each substrate-product pair
            for sub in substrates:
                for prod in products:
                    if user_labels is not None:
                        value = user_labels.get(reaction.name, 0)
                    else:
                        value = 0
                    compound_graph.add_edge(
                        sub, prod,
                        reaction=reaction.name,
                        value=value,
                        size=edge_sizes.get(reaction.name, 2) if edge_sizes is not None else 2
                    )

    # Prefer Graphviz when available, but keep plotting functional without
    # the optional pygraphviz dependency.
    pos = _graph_layout(compound_graph)
    label_nodes = pick_nodes_from_top_edges(compound_graph, top_k_edges=top_k_edges)
    
    # Scale edge sizes
    if edge_sizes is not None:
        subset_use_rxns = [r for r in use_rxns if r in edge_sizes.index]
        if subset_use_rxns:
            min_edge = edge_sizes[subset_use_rxns].min()
            max_edge = edge_sizes[subset_use_rxns].max()
            if max_edge - min_edge != 0:
                edge_sizes_scaled = (edge_sizes - min_edge) / (max_edge - min_edge) * (10 - 2) + 2
            else:
                edge_sizes_scaled = pd.Series(2, index=edge_sizes.index)
            for u, v, data in compound_graph.edges(data=True):
                r = data['reaction']
                compound_graph.edges[u, v]['size'] = edge_sizes_scaled.get(r, 2)
        else:
            for u, v in compound_graph.edges():
                compound_graph.edges[u, v]['size'] = 2
    else:
        for u, v in compound_graph.edges():
            compound_graph.edges[u, v]['size'] = 2

    # Calculate normalized values
    if user_labels is not None:
        subset_use_rxns = [r for r in use_rxns if r in user_labels.index]
        if subset_use_rxns:
            values = user_labels[subset_use_rxns]
            if edge_color_limits is not None:
                min_val, max_val = edge_color_limits
            else:
                max_abs = max(values.abs().max(), 1e-6)
                min_val = -max_abs
                max_val = max_abs
        else:
            if edge_color_limits is not None:
                min_val, max_val = edge_color_limits
            else:
                min_val = -1
                max_val = 1
    else:
        if edge_color_limits is not None:
            min_val, max_val = edge_color_limits
        else:
            min_val = -1
            max_val = 1
    
    # Create edge traces with midpoints for hover
    edge_traces = []
    mid_x, mid_y, edge_text = [], [], []
    
    for u, v, data in compound_graph.edges(data=True):
        norm_value = (data['value'] - min_val) / (max_val - min_val) if max_val != min_val else 0.5
        color = px.colors.sample_colorscale(color_scale, [norm_value])[0]
        
        # Edge line trace
        edge_traces.append(go.Scatter(
            x=[pos[u][0], pos[v][0], None],
            y=[pos[u][1], pos[v][1], None],
            line=dict(width=data['size'], color=color),
            mode='lines',
            hoverinfo='none',
            showlegend=False
        ))
        
        # Calculate midpoint for hover marker
        mid_x.append((pos[u][0] + pos[v][0])/2)
        mid_y.append((pos[u][1] + pos[v][1])/2)
        edge_text.append(f"Reaction: {data['reaction']}<br>Value: {data['value']:.2f}<br>Size: {data['size']:.2f}")

    # Create invisible hover markers trace
    hover_trace = go.Scatter(
        x=mid_x,
        y=mid_y,
        mode='markers',
        marker=dict(size=10, opacity=0),
        hoverinfo='text',
        text=edge_text,
        showlegend=False
    )

    # Get compound labels
    compound_labels = []
    node_colors = []
    hover_texts = []
    
    # Calculate node color normalization if node_labels provided
    node_min = None
    node_max = None
    if node_labels is not None:
        # Get node values for compounds that exist in the graph
        node_values = []
        for node in compound_graph.nodes():
            if node in node_labels.index:
                node_values.append(node_labels[node])
        
        if node_values:
            if node_color_limits is not None:
                node_min, node_max = node_color_limits
            else:
                max_abs_node = max(abs(max(node_values)), abs(min(node_values)), 1e-6)
                node_min = -max_abs_node
                node_max = max_abs_node
    
    for compound in compound_graph.nodes():
        if 'cpd:' in compound:
            s = compound.split('cpd:')
        elif 'gl:' in compound:
            s = compound.split('gl:')

        try:
            name = dataset.compound_info[s[1]]['name']
            compound_labels.append(name)
            hover_texts.append(f"Name: {name}<br>ID: {compound}")
        except KeyError:
            compound_labels.append(s[1])
            hover_texts.append(f"ID: {compound}")
            
        # Set node color based on normalized node_labels if provided
        if node_labels is not None and compound in node_labels.index and node_min is not None and node_max is not None:
            norm_value = (node_labels[compound] - node_min) / (node_max - node_min)
            node_colors.append(px.colors.sample_colorscale(node_color_scale, [norm_value])[0])
            hover_texts[-1] += f"<br>Value: {node_labels[compound]:.2f}"
        else:
            node_colors.append('lightgray')
            
        # Add size information to hover text if node_sizes provided
        if node_sizes is not None and compound in node_sizes.index:
            hover_texts[-1] += f"<br>Size: {node_sizes[compound]:.2f}"

    # Calculate node sizes if node_sizes provided
    node_size_values = []
    if node_sizes is not None:
        # Get node size values for compounds that exist in the graph
        size_values = []
        for node in compound_graph.nodes():
            if node in node_sizes.index:
                size_values.append(node_sizes[node])
        
        if size_values:
            min_size = min(size_values)
            max_size = max(size_values)
            # Scale sizes to range [12, 20] keeping minimum at 12 (original default)
            if max_size - min_size != 0:
                for node in compound_graph.nodes():
                    if node in node_sizes.index:
                        scaled_size = 12 + (node_sizes[node] - min_size) / (max_size - min_size) * (20 - 12)
                        node_size_values.append(scaled_size)
                    else:
                        node_size_values.append(12)  # Default size for missing values
            else:
                node_size_values = [12] * len(compound_graph.nodes())
        else:
            node_size_values = [12] * len(compound_graph.nodes())
    else:
        node_size_values = [12] * len(compound_graph.nodes())

    # Node trace
    text_out = []
    for compound, full_name in zip(compound_graph.nodes(), compound_labels):
        if compound in label_nodes:
            text_out.append(full_name[:16] + "…" if len(full_name) > 16 else full_name)
        else:
            text_out.append("")  # no label
            
    if show_node_labels:
        mode_type = 'markers+text'
    else:
        mode_type = 'markers'

    node_x = [pos[n][0] for n in compound_graph.nodes()]
    node_y = [pos[n][1] for n in compound_graph.nodes()]
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        text=text_out,
        mode=mode_type,
        hovertext=hover_texts,
        marker=dict(size=node_size_values, color=node_colors, opacity=1.0),
        hoverinfo='text',
        textposition='top center',
        hoverlabel=dict(bgcolor='white'),
        hovertemplate='%{hovertext}<extra></extra>'
    )

    # Create colorbar traces for both edges and nodes
    edge_colorbar = go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(
            colorscale=color_scale,
            showscale=True,
            cmin=min_val,
            cmax=max_val,
            colorbar=dict(
                title='Edge Value',
                thickness=15,
                len=0.5,
                yanchor='middle',
                y=0.5,
                xanchor='left',
                x=1.05
            )
        ),
        hoverinfo='none',
        showlegend=False
    )
    
    if node_labels is not None and node_min is not None and node_max is not None:
        node_colorbar = go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(
                colorscale=node_color_scale,
                showscale=True,
                cmin=node_min,
                cmax=node_max,
                colorbar=dict(
                    title='Node Value',
                    thickness=15,
                    len=0.5,
                    yanchor='middle',
                    y=0.5,
                    xanchor='left',
                    x=1.15
                )
            ),
            hoverinfo='none',
            showlegend=False
        )
        colorbar_traces = [edge_colorbar, node_colorbar]
    else:
        colorbar_traces = [edge_colorbar]
    
    # Assemble figure with correct layer order
    fig = go.Figure(data=edge_traces + [node_trace, hover_trace] + colorbar_traces)
    fig.update_layout(
        title='Custom Pathway Visualization',
        plot_bgcolor='white',
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        height=800
    )
    
    return fig

def rxn_volcano_plot(
    rxn_de_df: pd.DataFrame,
    module_ids: list = None,
    pathway_ids: list = None,
    prefix: str = None,
    cohens_d_col_suffix: str = "cohens_d",
    adj_pval_col_suffix: str = "adjusted_pval",
    xlim: tuple = None,
    ylim: tuple = None
):
    """
    Create a volcano plot for metabolic reactions, highlighting specific module or pathway ids.

    Parameters
    ----------
    rxn_de_df : pd.DataFrame
        Differential expression DataFrame for reactions. Should include module/pathway associations.
    module_ids : list of str, optional
        List of module IDs to highlight (column 'Modules' and 'Names'). Mutually exclusive with pathway_ids.
    pathway_ids : list of str, optional
        List of pathway IDs to highlight (column 'Pathways' and 'Pathway Names'). Mutually exclusive with module_ids.
    prefix : str, optional
        The prefix used to prefix columns for Cohen's d and adjusted p-value.
        If None, expects columns to be simply 'cohens_d' and 'adjusted_pval'.
    cohens_d_col_suffix : str, optional
        Suffix or full name for the Cohen's d column. Default is "cohens_d".
    adj_pval_col_suffix : str, optional
        Suffix or full name for the adjusted p-value column. Default is "adjusted_pval".
    xlim : tuple, optional
        Tuple specifying the (min, max) range for the x-axis.
    ylim : tuple, optional
        Tuple specifying the (min, max) range for the y-axis.
    """

    import matplotlib.pyplot as plt
    import numpy as np

    mode = None
    if (module_ids is not None) and (pathway_ids is not None):
        raise ValueError("Specify only one of module_ids or pathway_ids, not both.")
    elif module_ids is not None:
        mode = "module"
        ids = module_ids
        ids_col = "Modules"
        name_col = "Names"
    elif pathway_ids is not None:
        mode = "pathway"
        ids = pathway_ids
        ids_col = "Pathways"
        name_col = "Pathway Names"
    else:
        raise ValueError("You must specify one of module_ids or pathway_ids.")

    id_name_map = {}
    for item_id in ids:
        matching = rxn_de_df.loc[rxn_de_df[ids_col].str.contains(item_id, na=False)]
        if len(matching) > 0:
            temp = matching.iloc[0]
            split_names = temp[name_col].split(';')
            split_ids = temp[ids_col].split(';')
            for name, id in zip(split_names, split_ids):
                if id == item_id:
                    id_name_map[item_id] = name
                    break
            if item_id not in id_name_map:
                id_name_map[item_id] = item_id
        else:
            id_name_map[item_id] = item_id

    if prefix is None or prefix == "":
        cohens_d_col = cohens_d_col_suffix
        adj_pval_col = adj_pval_col_suffix
    else:
        cohens_d_col = f"{prefix}_{cohens_d_col_suffix}"
        adj_pval_col = f"{prefix}_{adj_pval_col_suffix}"

    plt.figure(figsize=(10, 8))

    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    n_highlight = len(ids)

    # Helper to filter out any (almost) gray color
    def is_gray(color, gray_tolerance=0.05, rgb_thresh=0.75):
        rgb = mcolors.to_rgb(color)
        # Allow close enough RGB values for gray, and exclude those with high brightness (to skip white)
        if abs(rgb[0] - rgb[1]) < gray_tolerance and abs(rgb[1] - rgb[2]) < gray_tolerance and abs(rgb[0] - rgb[2]) < gray_tolerance:
            if min(rgb) > rgb_thresh:  # Exclude white-ish also
                return True
            return True
        return False

    # Construct color list for highlights, explicitly skipping gray from tab10/tab20
    if n_highlight <= 10:
        cmap_colors = list(plt.get_cmap('tab10').colors)
        # In matplotlib tab10, gray is index 7: (0.498, 0.498, 0.498)
        gray_index_tab10 = 7
        # Remove all colors that are gray
        color_list = [c for i, c in enumerate(cmap_colors) if not is_gray(c)]
    elif n_highlight <= 20:
        cmap_colors = list(plt.get_cmap('tab20').colors)
        # In matplotlib tab20, gray is indices 7 and 15: (0.6, 0.6, 0.6), (0.749, 0.749, 0.749)
        gray_indices_tab20 = [7, 15]
        color_list = [c for i, c in enumerate(cmap_colors) if not is_gray(c)]
    else:
        # For more, just create from hsv ("rainbow"), which has no gray, or filter for grayish if any
        cmap = plt.get_cmap('hsv')
        color_list = [cmap(i / n_highlight) for i in range(n_highlight)]
        # In rare case hsv step is gray (shouldn't happen), filter
        color_list = [c for c in color_list if not is_gray(c)]

    # If not enough (removing gray reduced below n_highlight), repeat/warn user
    if len(color_list) < n_highlight:
        import warnings
        warnings.warn(
            f"Number of highlightable colors after excluding gray is less than number of ids ({len(color_list)} < {n_highlight}); will repeat some colors.",
            UserWarning
        )
        color_list = (color_list * ((n_highlight // len(color_list)) + 1))[:n_highlight]
    elif len(color_list) > n_highlight:
        color_list = color_list[:n_highlight]

    # Make a "master mask" for all highlighted ids,
    # so that no labeled modules or pathways end up in the "Other" group
    highlight_mask_any = rxn_de_df[ids_col].str.contains('|'.join(ids), na=False)
    mask_other = ~highlight_mask_any

    # Plot points NOT in highlighted modules/pathways
    plt.scatter(
        rxn_de_df.loc[mask_other, cohens_d_col],
        -np.log10(rxn_de_df.loc[mask_other, adj_pval_col]),
        alpha=0.5,
        color='gray',
        label='Other'
    )

    # Plot points IN highlighted modules/pathways, assigning new color to each
    for i, item_id in enumerate(ids):
        readable_name = id_name_map.get(item_id, item_id)
        highlight_mask = rxn_de_df[ids_col].str.contains(item_id, na=False)
        # Make sure to only include data not already labeled as another highlight (in case of overlap)
        mask_this = highlight_mask & highlight_mask_any
        plt.scatter(
            rxn_de_df.loc[mask_this, cohens_d_col],
            -np.log10(rxn_de_df.loc[mask_this, adj_pval_col]),
            alpha=0.7,
            color=color_list[i],
            label=readable_name
        )

    # Labels and formatting
    plt.xlabel("Cohen's d")
    plt.ylabel('-log10(Adjusted p-value)')
    plt.title('Volcano Plot of Metabolic Reactions')
    plt.axhline(y=-np.log10(0.05), color='r', linestyle='--', alpha=0.3)
    plt.legend()

    # Apply user-defined x and y limits if provided
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

def reaction_scatter_plot(
    rxn_de_df: pd.DataFrame,
    x: str,
    y: str,
    module_ids: list = None,
    pathway_ids: list = None,
    xlim: tuple = None,
    ylim: tuple = None,
    highlight_label_col: str = None,
    ax=None,
    figsize=(10, 8),
    scatter_kwargs=None
):
    """
    Create a scatter plot for reactions (e.g., volcano, PCA, or other),
    highlighting specific module or pathway ids if desired.

    Parameters
    ----------
    rxn_de_df : pd.DataFrame
        DataFrame containing reaction data.
    x : str
        Name of column to plot on the x-axis.
    y : str
        Name of column to plot on the y-axis.
    module_ids : list of str, optional
        List of module IDs to highlight (column 'Modules' and 'Names'). Mutually exclusive with pathway_ids.
    pathway_ids : list of str, optional
        List of pathway IDs to highlight (column 'Pathways' and 'Pathway Names'). Mutually exclusive with module_ids.
    xlim : tuple, optional
        Limits for x-axis.
    ylim : tuple, optional
        Limits for y-axis.
    highlight_label_col : str, optional
        Column which contains readable names for plotting the legend.
    ax : matplotlib.axes.Axes, optional
        Axes to plot to. If not provided, a new figure/axes is created.
    figsize : tuple, optional
        Size of the figure if ax=None.
    scatter_kwargs : dict, optional
        Additional scatter plot keyword arguments.
    """

    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    scatter_kwargs = scatter_kwargs or {}

    # Parameter validation for highlighting
    mode = None
    if (module_ids is not None) and (pathway_ids is not None):
        raise ValueError("Specify only one of module_ids or pathway_ids, not both.")
    elif module_ids is not None:
        mode = "module"
        ids = module_ids
        ids_col = "Modules"
        name_col = "Names"
    elif pathway_ids is not None:
        mode = "pathway"
        ids = pathway_ids
        ids_col = "Pathways"
        name_col = "Pathway Names"
    else:
        ids = []
        ids_col = None
        name_col = None

    id_name_map = {}
    if mode is not None:
        for item_id in ids:
            matching = rxn_de_df.loc[rxn_de_df[ids_col].str.contains(item_id, na=False)]
            if len(matching) > 0:
                temp = matching.iloc[0]
                split_names = temp[name_col].split(';')
                split_ids = temp[ids_col].split(';')
                for name, id in zip(split_names, split_ids):
                    if id == item_id:
                        id_name_map[item_id] = name
                        break
                if item_id not in id_name_map:
                    id_name_map[item_id] = item_id
            else:
                id_name_map[item_id] = item_id

    # Helper to filter out gray colors
    def is_gray(color, gray_tolerance=0.05, rgb_thresh=0.75):
        rgb = mcolors.to_rgb(color)
        if abs(rgb[0] - rgb[1]) < gray_tolerance and abs(rgb[1] - rgb[2]) < gray_tolerance and abs(rgb[0] - rgb[2]) < gray_tolerance:
            if min(rgb) > rgb_thresh:  # Exclude white-ish also
                return True
            return True
        return False

    n_highlight = len(ids)
    # Prepare color list for highlights
    if n_highlight <= 10:
        cmap_colors = list(plt.get_cmap('tab10').colors)
        color_list = [c for i, c in enumerate(cmap_colors) if not is_gray(c)]
    elif n_highlight <= 20:
        cmap_colors = list(plt.get_cmap('tab20').colors)
        color_list = [c for i, c in enumerate(cmap_colors) if not is_gray(c)]
    else:
        cmap = plt.get_cmap('hsv')
        color_list = [cmap(i / n_highlight) for i in range(n_highlight)]
        color_list = [c for c in color_list if not is_gray(c)]

    # If not enough (after removing gray), repeat as needed
    if len(color_list) < n_highlight and n_highlight > 0:
        import warnings
        warnings.warn(
            f"Number of highlightable colors after excluding gray is less than number of ids ({len(color_list)} < {n_highlight}); will repeat some colors.",
            UserWarning
        )
        color_list = (color_list * ((n_highlight // len(color_list)) + 1))[:n_highlight]
    elif len(color_list) > n_highlight:
        color_list = color_list[:n_highlight]

    # Make a mask for highlighted ids, if any, so "Other" doesn't get double-labeled
    if mode is not None:
        highlight_mask_any = rxn_de_df[ids_col].str.contains('|'.join(ids), na=False)
        mask_other = ~highlight_mask_any
    else:
        highlight_mask_any = None
        mask_other = None

    # Plot non-highlighted points as gray
    if mask_other is not None:
        ax.scatter(
            rxn_de_df.loc[mask_other, x],
            rxn_de_df.loc[mask_other, y],
            alpha=0.5,
            color='gray',
            label='Other',
            **scatter_kwargs
        )
    else:
        # If no highlight, plot all as gray
        ax.scatter(
            rxn_de_df[x], rxn_de_df[y],
            alpha=0.6,
            color='gray',
            label='All',
            **scatter_kwargs
        )

    # Plot highlighted points in color, one per id
    if mode is not None and n_highlight > 0:
        for i, item_id in enumerate(ids):
            readable_name = id_name_map.get(item_id, item_id)
            highlight_mask = rxn_de_df[ids_col].str.contains(item_id, na=False)
            mask_this = highlight_mask & highlight_mask_any
            if highlight_label_col is not None and highlight_label_col in rxn_de_df.columns:
                # Use highlight_label_col for legend label (e.g. more finegrained)
                legend_label = rxn_de_df.loc[mask_this, highlight_label_col].iloc[0] \
                    if (~rxn_de_df.loc[mask_this, highlight_label_col].isna()).any() else readable_name
            else:
                legend_label = readable_name
            ax.scatter(
                rxn_de_df.loc[mask_this, x],
                rxn_de_df.loc[mask_this, y],
                alpha=0.7,
                color=color_list[i],
                label=legend_label,
                **scatter_kwargs
            )

    # Labels and formatting
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title('Reaction Scatter Plot')
    ax.legend()

    # Axis limits
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    return ax
