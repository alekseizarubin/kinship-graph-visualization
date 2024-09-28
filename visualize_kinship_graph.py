import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

def parse_arguments():
    parser = argparse.ArgumentParser(description="Visualize kinship groups from PLINK genome file as a graph.")
    parser.add_argument("genome_file", help="Path to the PLINK genome file.")
    parser.add_argument("--output", help="Prefix for output graph image files.", default="kinship_graph")
    parser.add_argument("--dpi", type=int, help="Resolution of the output image in dots per inch (DPI).", default=100)
    parser.add_argument("--figsize", type=float, nargs=2, help="Figure size in inches (width height).", default=[10, 8])
    parser.add_argument("--fontsize", type=int, help="Font size for labels.", default=12)
    parser.add_argument("--edge-width", type=float, help="Width of the edges in the graph.", default=2)
    parser.add_argument("--node-size", type=int, help="Size of the nodes in the graph.", default=500)
    parser.add_argument("--format", choices=['png', 'tiff', 'jpeg'], help="Format to save the output image.", default='png')
    parser.add_argument("--max-degree-fraction", type=float, help="Maximum allowed fraction of connections per sample. Samples with higher fraction will be excluded.", default=None)
    parser.add_argument("--threshold1", type=float, help="PI_HAT threshold for color 1 (default 0.75).", default=0.75)
    parser.add_argument("--threshold2", type=float, help="PI_HAT lower threshold for colors 2 and 3 (default 0.4).", default=0.4)
    parser.add_argument("--z1-threshold", type=float, help="Z1 threshold for distinguishing between colors 2 and 3 (default 0.75).", default=0.75)
    parser.add_argument("--haplogroup-Y", help="Path to the file containing Y haplogroup information.", default=None)
    parser.add_argument("--haplogroup-MT", help="Path to the file containing MT haplogroup information.", default=None)
    parser.add_argument("--legend", action='store_true', help="Add legend to the image.")
    return parser.parse_args()

def read_genome_file(genome_file):
    """Reads the PLINK genome file and preprocesses the data."""
    if not os.path.exists(genome_file):
        raise FileNotFoundError(f"The file {genome_file} does not exist.")
    # Read the file and ensure IID1 and IID2 are strings without leading/trailing spaces
    genome_df = pd.read_csv(genome_file, sep='\s+', dtype={'IID1': str, 'IID2': str})
    genome_df['IID1'] = genome_df['IID1'].str.strip()
    genome_df['IID2'] = genome_df['IID2'].str.strip()
    return genome_df

def read_haplogroup_file(haplogroup_file):
    """Reads the haplogroup file and preprocesses the data."""
    if not os.path.exists(haplogroup_file):
        raise FileNotFoundError(f"The file {haplogroup_file} does not exist.")
    # Read only the first two columns (Sample and Haplogroup)
    haplo_df = pd.read_csv(haplogroup_file, sep='\t', header=None, names=['Sample', 'Haplogroup'], usecols=[0, 1], dtype={'Sample': str})
    haplo_df['Sample'] = haplo_df['Sample'].str.strip()
    haplo_df['Haplogroup'] = haplo_df['Haplogroup'].astype(str).str.strip()
    return haplo_df

def create_kinship_graph(genome_df, max_degree_fraction=None, thresholds=None):
    """Creates the kinship graph based on PI_HAT and Z1 thresholds."""
    G = nx.Graph()
    
    # Get all unique samples from IID1 and IID2
    samples_iid1 = set(genome_df['IID1'].unique())
    samples_iid2 = set(genome_df['IID2'].unique())
    all_samples = samples_iid1.union(samples_iid2)
    total_samples = len(all_samples)
    
    # Add all nodes to the graph
    for sample in all_samples:
        G.add_node(sample)
    
    # Set thresholds for edge colors
    threshold1 = thresholds.get('threshold1', 0.75)
    threshold2 = thresholds.get('threshold2', 0.4)
    z1_threshold = thresholds.get('z1_threshold', 0.75)
    
    # Add edges based on PI_HAT and Z1 values
    for _, row in genome_df.iterrows():
        sample1 = row['IID1']
        sample2 = row['IID2']
        pi_hat = row['PI_HAT']
        z1 = row['Z1']
        
        # Determine edge color based on thresholds
        if pi_hat > threshold1:
            edge_color = 'red'  # Color 1
        elif threshold2 <= pi_hat <= threshold1 and z1 > z1_threshold:
            edge_color = 'blue'  # Color 2
        elif threshold2 <= pi_hat <= threshold1 and z1 <= z1_threshold:
            edge_color = 'green'  # Color 3
        else:
            edge_color = 'gray'  # Color 4
        
        # Add edge to the graph if PI_HAT > 0
        if pi_hat > 0:
            G.add_edge(sample1, sample2, weight=pi_hat, color=edge_color)
    
    # Apply filter to remove nodes with a high degree
    if max_degree_fraction is not None:
        max_degree = max_degree_fraction * total_samples
        nodes_to_remove = [node for node, degree in G.degree() if degree > max_degree]
        if nodes_to_remove:
            print(f"Removing {len(nodes_to_remove)} samples with degree higher than {max_degree:.2f} ({max_degree_fraction*100}% of total samples).")
            print("Samples removed due to high degree:")
            for node in nodes_to_remove:
                print(f"- {node}")
            G.remove_nodes_from(nodes_to_remove)
        else:
            print("No samples exceed the maximum degree fraction.")
    
    return G

def remove_isolated_nodes(G):
    """Removes isolated nodes (nodes with no connections) from the graph."""
    isolated_nodes = list(nx.isolates(G))
    if isolated_nodes:
        print(f"Removing {len(isolated_nodes)} isolated samples with no connections.")
        G.remove_nodes_from(isolated_nodes)
    else:
        print("No isolated samples to remove.")
    return G

def assign_node_colors(G, haplogroup_Y_file=None, haplogroup_MT_file=None):
    """Assigns colors to nodes based on haplogroup information."""
    node_colors = {}
    haplogroup_colors = {}
    color_palette = plt.cm.tab20.colors  # Use 20 colors from the palette

    # Color index counter
    color_index = 0

    # Load and process Y haplogroups
    haplo_Y = {}
    if haplogroup_Y_file:
        haplo_Y_df = read_haplogroup_file(haplogroup_Y_file)
        for _, row in haplo_Y_df.iterrows():
            sample = str(row['Sample']).strip()
            haplogroup = str(row['Haplogroup']).strip() if pd.notnull(row['Haplogroup']) else 'Unknown_Y'
            haplo_Y[sample] = haplogroup
        haplo_Y_unknown = set(G.nodes()) - set(haplo_Y.keys())
        for sample in haplo_Y_unknown:
            haplo_Y[sample] = 'Unknown_Y'
        # For debugging
        common_samples_Y = set(haplo_Y.keys()).intersection(set(G.nodes()))
        print(f"Y haplogroup info: {len(common_samples_Y)} samples matched out of {len(G.nodes())}")

    # Load and process MT haplogroups
    haplo_MT = {}
    if haplogroup_MT_file:
        haplo_MT_df = read_haplogroup_file(haplogroup_MT_file)
        for _, row in haplo_MT_df.iterrows():
            sample = str(row['Sample']).strip()
            haplogroup = str(row['Haplogroup']).strip() if pd.notnull(row['Haplogroup']) else 'Unknown_MT'
            haplo_MT[sample] = haplogroup
        haplo_MT_unknown = set(G.nodes()) - set(haplo_MT.keys())
        for sample in haplo_MT_unknown:
            haplo_MT[sample] = 'Unknown_MT'
        # For debugging
        common_samples_MT = set(haplo_MT.keys()).intersection(set(G.nodes()))
        print(f"MT haplogroup info: {len(common_samples_MT)} samples matched out of {len(G.nodes())}")

    # Create a set of haplogroups and assign colors
    haplogroup_set = set()
    if haplogroup_Y_file:
        haplogroup_set.update(set(haplo_Y.values()))
    if haplogroup_MT_file:
        haplogroup_set.update(set(haplo_MT.values()))
    haplogroup_set.discard('Unknown_Y')
    haplogroup_set.discard('Unknown_MT')
    haplogroup_list = sorted(list(haplogroup_set))
    for haplogroup in haplogroup_list:
        haplogroup_colors[haplogroup] = color_palette[color_index % len(color_palette)]
        color_index += 1
    # Add colors for unknown haplogroups
    if haplogroup_Y_file:
        haplogroup_colors['Unknown_Y'] = (0.5, 0.5, 0.5)  # Gray
    if haplogroup_MT_file:
        haplogroup_colors['Unknown_MT'] = (0.7, 0.7, 0.7)  # Light gray

    # Assign colors to nodes
    for node in G.nodes():
        node_info = {}
        if haplogroup_Y_file:
            hg_Y = haplo_Y.get(node, 'Unknown_Y')
            color_Y = haplogroup_colors.get(hg_Y, (0.5, 0.5, 0.5))
            node_info['Y_haplogroup'] = hg_Y
            node_info['Y_color'] = color_Y
        if haplogroup_MT_file:
            hg_MT = haplo_MT.get(node, 'Unknown_MT')
            color_MT = haplogroup_colors.get(hg_MT, (0.7, 0.7, 0.7))
            node_info['MT_haplogroup'] = hg_MT
            node_info['MT_color'] = color_MT
        if node_info:
            node_colors[node] = node_info
        else:
            # No haplogroup information
            node_colors[node] = {'color': (0.8, 0.8, 0.8)}
    return node_colors, haplogroup_colors

def visualize_graph_component(G, component, component_id, output_prefix, dpi=100, figsize=(10, 8), fontsize=12,
                              edge_width=2, node_size=500, image_format='png', node_colors=None, haplogroup_colors=None,
                              show_legend=False, thresholds=None):
    """Visualizes a connected component of the graph."""
    # Set font size
    plt.rcParams.update({'font.size': fontsize})
    
    # Create a subgraph for the component
    subgraph = G.subgraph(component).copy()
    
    # Skip visualization if the subgraph has no edges
    if subgraph.number_of_edges() == 0:
        print(f"Component {component_id} has no edges. Skipping visualization.")
        return
    
    # Get edge colors and labels (PI_HAT)
    edge_colors = nx.get_edge_attributes(subgraph, 'color')
    edge_labels = nx.get_edge_attributes(subgraph, 'weight')
    
    # Positions for nodes
    pos = nx.spring_layout(subgraph, seed=42)
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Draw edges
    nx.draw_networkx_edges(subgraph, pos, edge_color=list(edge_colors.values()), width=edge_width, ax=ax)
    
    # Draw nodes with haplogroup colors
    if node_colors:
        node_sample = list(node_colors.values())[0]
        if 'Y_haplogroup' in node_sample and 'MT_haplogroup' in node_sample:
            # Both Y and MT haplogroups provided
            for node in subgraph.nodes():
                node_info = node_colors[node]
                color_MT = node_info['MT_color']
                color_Y = node_info['Y_color']
                nx.draw_networkx_nodes(subgraph, pos, nodelist=[node], node_size=node_size, node_color=[color_MT], edgecolors=[color_Y], linewidths=2, ax=ax)
        elif 'Y_haplogroup' in node_sample:
            # Only Y haplogroups provided
            colors = [node_colors[node]['Y_color'] for node in subgraph.nodes()]
            nx.draw_networkx_nodes(subgraph, pos, node_color=colors, node_size=node_size, ax=ax)
        elif 'MT_haplogroup' in node_sample:
            # Only MT haplogroups provided
            colors = [node_colors[node]['MT_color'] for node in subgraph.nodes()]
            nx.draw_networkx_nodes(subgraph, pos, node_color=colors, node_size=node_size, ax=ax)
        else:
            # No haplogroup information
            nx.draw_networkx_nodes(subgraph, pos, node_color='lightgray', node_size=node_size, ax=ax)
    else:
        # No haplogroup information
        nx.draw_networkx_nodes(subgraph, pos, node_color='lightgray', node_size=node_size, ax=ax)
    
    # Draw node labels
    labels = {node: node for node in subgraph.nodes()}  # Use IID as label
    nx.draw_networkx_labels(subgraph, pos, labels=labels, ax=ax)
    
    # Draw edge labels (PI_HAT values)
    edge_labels_formatted = {edge: f"{edge_labels[edge]:.2f}" for edge in edge_labels}
    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels_formatted, ax=ax)
    
    ax.set_title(f'Kinship Graph Component {component_id} with PI_HAT values')
    ax.axis('off')  # Hide axes
    
    # Add legend if required
    if show_legend and thresholds:
        handles = []
        # Legend for edge colors
        edge_color_map = {
            'red': f'PI_HAT > {thresholds["threshold1"]}',
            'blue': f'{thresholds["threshold2"]} ≤ PI_HAT ≤ {thresholds["threshold1"]}, Z1 > {thresholds["z1_threshold"]}',
            'green': f'{thresholds["threshold2"]} ≤ PI_HAT ≤ {thresholds["threshold1"]}, Z1 ≤ {thresholds["z1_threshold"]}',
            'gray': f'PI_HAT < {thresholds["threshold2"]}'
        }
        for color, desc in edge_color_map.items():
            handles.append(Line2D([0], [0], color=color, lw=2, label=desc))
        
        # Legend for haplogroups
        if haplogroup_colors:
            haplogroups_in_subgraph_MT = set()
            haplogroups_in_subgraph_Y = set()
            node_sample = list(node_colors.values())[0]
            if 'Y_haplogroup' in node_sample and 'MT_haplogroup' in node_sample:
                # Both Y and MT haplogroups provided
                for node in subgraph.nodes():
                    node_info = node_colors[node]
                    hg_MT = node_info['MT_haplogroup']
                    hg_Y = node_info['Y_haplogroup']
                    haplogroups_in_subgraph_MT.add(hg_MT)
                    haplogroups_in_subgraph_Y.add(hg_Y)
                # Legend for MT haplogroups
                for hg in haplogroups_in_subgraph_MT:
                    color = haplogroup_colors[hg]
                    label = f'MT Haplogroup {hg}' if not hg.startswith('Unknown') else f'{hg}'
                    handles.append(Patch(facecolor=color, edgecolor='black', label=label))
                # Legend for Y haplogroups
                for hg in haplogroups_in_subgraph_Y:
                    color = haplogroup_colors[hg]
                    label = f'Y Haplogroup {hg}' if not hg.startswith('Unknown') else f'{hg}'
                    handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markeredgecolor=color, markersize=10, linewidth=0, label=label))
            elif 'Y_haplogroup' in node_sample:
                # Only Y haplogroups
                haplogroups_in_subgraph = set()
                for node in subgraph.nodes():
                    node_info = node_colors[node]
                    hg = node_info['Y_haplogroup']
                    haplogroups_in_subgraph.add(hg)
                for hg in haplogroups_in_subgraph:
                    color = haplogroup_colors[hg]
                    label = f'Haplogroup {hg}' if not hg.startswith('Unknown') else f'{hg}'
                    handles.append(Patch(facecolor=color, edgecolor='black', label=label))
            elif 'MT_haplogroup' in node_sample:
                # Only MT haplogroups
                haplogroups_in_subgraph = set()
                for node in subgraph.nodes():
                    node_info = node_colors[node]
                    hg = node_info['MT_haplogroup']
                    haplogroups_in_subgraph.add(hg)
                for hg in haplogroups_in_subgraph:
                    color = haplogroup_colors[hg]
                    label = f'Haplogroup {hg}' if not hg.startswith('Unknown') else f'{hg}'
                    handles.append(Patch(facecolor=color, edgecolor='black', label=label))
        # Place legend outside the plot area
        ax.legend(handles=handles, fontsize=fontsize, loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    
    # Adjust layout to make room for the legend
    plt.tight_layout()
    
    # Save the image
    output_file = f"{output_prefix}_component_{component_id}.{image_format}"
    plt.savefig(output_file, dpi=dpi, format=image_format, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory
    print(f"Graph component {component_id} saved to {output_file}")

def main():
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set thresholds for edge colors
    thresholds = {
        'threshold1': args.threshold1,
        'threshold2': args.threshold2,
        'z1_threshold': args.z1_threshold
    }

    # Read genome file
    genome_df = read_genome_file(args.genome_file)
    
    # Create kinship graph
    G = create_kinship_graph(genome_df, max_degree_fraction=args.max_degree_fraction, thresholds=thresholds)
    
    # Remove isolated nodes
    G = remove_isolated_nodes(G)
    
    # Assign colors to nodes based on haplogroups
    node_colors, haplogroup_colors = assign_node_colors(
        G,
        haplogroup_Y_file=args.haplogroup_Y,
        haplogroup_MT_file=args.haplogroup_MT
    )
    
    # Output graph information
    print(f"Total samples (nodes): {G.number_of_nodes()}")
    print(f"Total relationships (edges): {G.number_of_edges()}")
    
    # Get connected components
    components = list(nx.connected_components(G))
    
    # Visualize each component
    for i, component in enumerate(components, start=1):
        visualize_graph_component(
            G,
            component,
            i,
            args.output,
            dpi=args.dpi,
            figsize=args.figsize,
            fontsize=args.fontsize,
            edge_width=args.edge_width,
            node_size=args.node_size,
            image_format=args.format,
            node_colors=node_colors,
            haplogroup_colors=haplogroup_colors,
            show_legend=args.legend,
            thresholds=thresholds
        )

if __name__ == "__main__":
    main()
