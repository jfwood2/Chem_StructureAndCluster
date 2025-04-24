import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Draw
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import pairwise_distances
from sklearn.manifold import SpectralEmbedding
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
import umap
import matplotlib.patches as mpatches
import os

# Load SMILES from file
def load_smiles(file_path):
    with open(file_path, 'r') as f:
        smiles_list = [line.strip() for line in f.readlines()]
    return smiles_list

# Load drug names from file
def load_drug_names(drug_file_path):
    with open(drug_file_path, 'r') as f:
        drug_names = [line.strip() for line in f.readlines()]
    return drug_names

# Convert SMILES to RDKit molecules
def smiles_to_molecules(smiles_list):
    mols = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"Invalid SMILES skipped: {smi}")
        mols.append(mol)
    return mols

# Compute fingerprints
def compute_fingerprints(molecules, radius=2, n_bits=1024):
    fps = []
    for mol in molecules:
        if mol is None:
            fps.append(None)
        else:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits))
    return fps


# Compute similarity matrix (intra-group)
def compute_similarity_matrix(fingerprints):
    num_mols = len(fingerprints)
    sim_matrix = np.zeros((num_mols, num_mols))
    for i in range(num_mols):
        for j in range(num_mols):
            sim_matrix[i, j] = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
    return sim_matrix

# Compute cross-group similarity matrix (inter-group)
def compute_cross_similarity_matrix(fps1, fps2):
    sim_matrix = np.zeros((len(fps1), len(fps2)))
    for i in range(len(fps1)):
        for j in range(len(fps2)):
            sim_matrix[i, j] = DataStructs.TanimotoSimilarity(fps1[i], fps2[j])
    return sim_matrix
 

# Plot heatmap with clustering
# Improved heatmap with colorbar on right and labels on left/bottom
def plot_clustered_heatmap(sim_matrix, row_labels, col_labels, output_path, filename="clustered_similarity_heatmap.png", title="Clustered Tanimoto Similarity Heatmap"):
    # Perform hierarchical clustering
    linkage_rows = linkage(sim_matrix, method='average')
    linkage_cols = linkage(sim_matrix.T, method='average')
    row_order = leaves_list(linkage_rows)
    col_order = leaves_list(linkage_cols)

    # Reorder matrix and labels accordingly
    ordered_matrix = sim_matrix[row_order][:, col_order]
    ordered_row_labels = [row_labels[i] for i in row_order]
    ordered_col_labels = [col_labels[i] for i in col_order]

    # Plot heatmap
    plt.figure(figsize=(12, 10))

    # Create a grid for the heatmap and dendrograms
    gs = plt.GridSpec(2, 2, width_ratios=[1, 8], height_ratios=[1, 8]) 
    ax_dendrogram_row = plt.subplot(gs[0, 1])  # Dendrogram for rows
    ax_dendrogram_col = plt.subplot(gs[1, 0])  # Dendrogram for columns
    ax_heatmap = plt.subplot(gs[1, 1])        # Heatmap area

    # Plot dendrograms
    dendrogram(linkage_rows, ax=ax_dendrogram_row, orientation='top', no_labels=True, color_threshold=0, above_threshold_color='black')
    dendrogram(linkage_cols, ax=ax_dendrogram_col, orientation='right', no_labels=True, color_threshold=0, above_threshold_color='black')

    for ax in [ax_dendrogram_row, ax_dendrogram_col]:
            ax.set_facecolor('none')  # Remove background color
            for spine in ax.spines.values():
                spine.set_visible(False)  # Hide the spines (box)

    # Hide axes for dendrograms
    ax_dendrogram_row.set_xticks([])
    ax_dendrogram_row.set_yticks([])
    ax_dendrogram_col.set_xticks([])
    ax_dendrogram_col.set_yticks([])

    # Plot the heatmap in the specified area
    sns.heatmap(
        ordered_matrix,
        xticklabels=ordered_col_labels,
        yticklabels=ordered_row_labels,
        cmap='Blues',
        annot=True,
        fmt=".2f",
        annot_kws={"size": 5.7},
        cbar_kws={"label": "Tanimoto Similarity"},
        linewidths=0.3,
        square=True,
        ax=ax_heatmap
    )

    ax_heatmap.set_xticklabels(ax_heatmap.get_xticklabels(), rotation=90)
    ax_heatmap.set_yticklabels(ax_heatmap.get_yticklabels(), rotation=0)
    ax_heatmap.set_title(title, loc='center', fontsize=12, pad=10)
    


    # Tight layout for better spacing
    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(output_path, filename), dpi=1200)
    plt.close()


# UMAP plot with improved layout and optional labels
def plot_umap_with_clustering_after(fingerprints, labels, output_path, filename="umap_plot.png", drugs_to_label=None, clustering_method="Kmeans", n_clusters=5, n_neighbors=10, min_dist=0.04):
    dot_size = 350
    # Convert fingerprints into a format UMAP can use
    fp_array = np.array([list(fp) for fp in fingerprints])

    # Perform UMAP transformation 
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    embeddings = reducer.fit_transform(fp_array)

    # Apply clustering after UMAP transformation
    if clustering_method == "KMeans":
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    else:
        clusterer = AgglomerativeClustering(n_clusters=n_clusters)

    cluster_labels = clusterer.fit_predict(embeddings)

    # Define a consistent colormap (similar to heatmap color)
    cmap = plt.get_cmap('tab20')  # Use the same colormap as the heatmap

    # Plot UMAP with clustering
    plt.figure(figsize=(12, 9))

    # Scatter plot colored by cluster labels
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=cluster_labels, s=dot_size, alpha=0.7, cmap=cmap, edgecolors='none', linewidths=0.4)

    # Optionally, add specific drug labels (e.g., Neratinib, Pelitinib)
    if drugs_to_label:
        for i, label in enumerate(labels):
            if label in drugs_to_label:  # Only label drugs that exist in this plot
                plt.text(embeddings[i, 0], embeddings[i, 1] + 0.15, label, fontsize=7, ha='center', va='bottom', color='black')

    # Add colorbar to UMAP with consistent labeling
    legend_labels = [f"Cluster #{i+1}" for i in range(n_clusters)]
    patches = [mpatches.Patch(color=cmap(i / (n_clusters - 1)), label=legend_labels[i]) for i in range(n_clusters)]
    plt.legend(handles=patches, loc='upper right', fontsize=10)




    # Remove unnecessary gridlines and improve layout
    plt.grid(False)  # Remove gridlines
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    
    # Set title and improve readability
    plt.title("UMAP Representation of Kmeans Clustering", fontsize=12, loc='center', pad=10)
    
    # Tight layout for better spacing
    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(output_path, filename), dpi=600)
    plt.close()



# Main
def main():
    smiles_file_1 = "/Users/jakewood/Downloads/UTCOMLS/UTCOMLS Research/Dr. Robert Smith/JW_Drug_Sim_Calc/test_2Day_SMILES.txt"
    names_file_1 = "/Users/jakewood/Downloads/UTCOMLS/UTCOMLS Research/Dr. Robert Smith/JW_Drug_Sim_Calc/test_2Day_drug_names.txt"
    smiles_file_2 = "/Users/jakewood/Downloads/UTCOMLS/UTCOMLS Research/Dr. Robert Smith/JW_Drug_Sim_Calc/test_5Day_SMILES.txt"
    names_file_2 = "/Users/jakewood/Downloads/UTCOMLS/UTCOMLS Research/Dr. Robert Smith/JW_Drug_Sim_Calc/Test_5Day_drug_names.txt"
    drugs_to_label = ["Neratinib", "Pelitinib", "Erlotinib", "Tyrphostin_AG-1478"]
    output_path = "/Users/jakewood/Downloads/UTCOMLS/UTCOMLS Research/Dr. Robert Smith/JW_Drug_Sim_Calc/Output/"
    os.makedirs(output_path, exist_ok=True)

    smiles_1 = load_smiles(smiles_file_1)
    names_1 = load_drug_names(names_file_1)
    mols_1 = smiles_to_molecules(smiles_1)
    fps_1 = compute_fingerprints(mols_1)

    sim_matrix_1 = compute_similarity_matrix(fps_1)
    plot_clustered_heatmap(
        sim_matrix_1,
        row_labels=names_1,
        col_labels=names_1,
        output_path=output_path,
        filename="clustered_similarity_heatmap_2Day.png"
)
    plot_umap_with_clustering_after(fps_1, names_1, output_path, filename="umap_2Day.png", drugs_to_label=drugs_to_label, clustering_method="Kmeans")

    smiles_2 = load_smiles(smiles_file_2)
    names_2 = load_drug_names(names_file_2)
    mols_2 = smiles_to_molecules(smiles_2)
    fps_2 = compute_fingerprints(mols_2)

    sim_matrix_2 = compute_similarity_matrix(fps_2)
    plot_clustered_heatmap(
        sim_matrix_2,
        row_labels=names_2,
        col_labels=names_2,
        output_path=output_path,
        filename="clustered_similarity_heatmap_5Day.png"
)
    plot_umap_with_clustering_after(fps_2, names_2, output_path, filename="umap_5Day.png", drugs_to_label=drugs_to_label, clustering_method="Kmeans")

    cross_sim_matrix = compute_cross_similarity_matrix(fps_1, fps_2)
    plot_clustered_heatmap(
        cross_sim_matrix,
        row_labels=names_1,
        col_labels=names_2,
        output_path=output_path,
        filename="cross_clustered_similarity_heatmap.png",
        title="Clustered Tanimoto Similarity Heatmap (Cross-Group)"
)


    combined_fps = fps_1 + fps_2
    combined_names = names_1 + names_2
    plot_umap_with_clustering_after(combined_fps, combined_names, output_path, filename="umap_combined.png", drugs_to_label=drugs_to_label, clustering_method="Kmeans")

if __name__ == "__main__":
    main()