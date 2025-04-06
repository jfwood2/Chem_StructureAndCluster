import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Draw
from sklearn.manifold import TSNE
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
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
    return [Chem.MolFromSmiles(smiles) for smiles in smiles_list]


# Compute Morgan fingerprints
def compute_fingerprints(molecules, radius=2, n_bits=1024):
    return [AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits) for mol in molecules]


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


# Extract common scaffolds
def extract_scaffolds(molecules):
    return [MurckoScaffold.GetScaffoldForMol(mol) for mol in molecules]


# Cluster scaffolds based on similarity
def cluster_scaffolds(similarity_matrix, n_clusters=5):
    distance_matrix = 1 - similarity_matrix
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='average')
    return clustering.fit_predict(distance_matrix)


# Updated plot_heatmap function to support asymmetric matrices
def plot_heatmap(sim_matrix, row_labels, output_path, col_labels=None, filename="similarity_heatmap.png", title="Tanimoto Similarity Heatmap"):
    plt.figure(figsize=(12, 9))
    sns.heatmap(
        sim_matrix,
        xticklabels=col_labels if col_labels else row_labels,
        yticklabels=row_labels,
        annot=True,
        fmt=".2f",
        #Change the cmap for better visibility
        cmap='Blues',
        annot_kws={"size": 5.7},
        cbar_kws={"label": "Tanimoto Similarity"},
        linewidths=0.5,
        linecolor='black',
        square=False
    )
    plt.xticks(rotation=45, ha="right", fontsize=6.6)
    plt.yticks(rotation=0, fontsize=6.6)
    plt.title(title, fontsize=16)
    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(output_path, filename), dpi=600)
    plt.close()




# t-SNE plot
def plot_tsne_with_similarity(fingerprints, labels, output_path, filename="tsne_colored_by_similarity.png"):
    # Convert fingerprints to array if needed
    fp_array = np.array([list(fp) for fp in fingerprints])
    
    # Perform t-SNE to get 2D embeddings
    tsne = TSNE(n_components=2, random_state=42)
    perplexity = min(len(fingerprints) - 1, 2)  # Set perplexity to 2 or adjust based on the number of samples
    tsne.perplexity = perplexity
    embeddings = tsne.fit_transform(fp_array)
    
    # Compute pairwise Euclidean distance matrix on the t-SNE 2D embeddings
    distance_matrix = pairwise_distances(embeddings, metric='euclidean')
    
    # Color points based on the average distance to all other points
    avg_distances = np.mean(distance_matrix, axis=1)
    
    # Normalize distances to a 0-1 scale for coloring
    norm_distances = (avg_distances - np.min(avg_distances)) / (np.max(avg_distances) - np.min(avg_distances))
    
    # Create the plot
    plt.figure(figsize=(12, 9))
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=norm_distances, cmap='Blues', s=80, alpha=0.7)
    
    # Annotate points with the corresponding labels and the +.5 offset puts the labels above the points
    for i, txt in enumerate(labels):
        plt.annotate(txt, (embeddings[i, 0], embeddings[i, 1] + 0.25 ), fontsize=4, ha='center', va='bottom', alpha=0.7)
    

    # Title t-SNE plot
    plt.title("t-SNE Visualization")
    plt.colorbar(scatter, label='Average Distance to Other Points')
    plt.tight_layout(pad=2.0)

    # Save plot
    plt.savefig(os.path.join(output_path, filename), dpi=600)
    plt.close()






# Plot scaffolds
def plot_scaffolds(scaffolds, output_path):
    img = Draw.MolsToGridImage(scaffolds, molsPerRow=3, subImgSize=(200, 200))
    img.save(os.path.join(output_path, "scaffold_grid.png"))



# Main
def main():
    # === Paths ===
    smiles_file_1 = "/Users/jakewood/Downloads/UTCOMLS/UTCOMLS Research/Dr. Robert Smith/KimSin/JW_Sim_Calc/test_2Day_SMILES.txt"
    names_file_1 = "/Users/jakewood/Downloads/UTCOMLS/UTCOMLS Research/Dr. Robert Smith/KimSin/JW_Sim_Calc/test_2Day_drug_names.txt"
    smiles_file_2 = "/Users/jakewood/Downloads/UTCOMLS/UTCOMLS Research/Dr. Robert Smith/KimSin/JW_Sim_Calc/test_5Day_SMILES.txt"
    names_file_2 = "/Users/jakewood/Downloads/UTCOMLS/UTCOMLS Research/Dr. Robert Smith/KimSin/JW_Sim_Calc/test_5Day_drug_names.txt"
    
    #Where to save the output!!
    output_path = "/Users/jakewood/Downloads/UTCOMLS/UTCOMLS Research/Dr. Robert Smith/KimSin/JW_Sim_Calc/output"
    

    os.makedirs(output_path, exist_ok=True)

    # === Load group 1 ===
    smiles_1 = load_smiles(smiles_file_1)
    names_1 = load_drug_names(names_file_1)
    mols_1 = smiles_to_molecules(smiles_1)
    fps_1 = compute_fingerprints(mols_1)

    # === Single-group comparison for group 1 ===
    sim_matrix = compute_similarity_matrix(fps_1)
    plot_heatmap(sim_matrix, names_1, output_path)

    # Optional: t-SNE just for group 1
    plot_tsne_with_similarity(fps_1, names_1, output_path, filename="tsne_2Day.png")
    
    
    # commented out the clustering part for now, but you can uncomment it if needed
    '''
    # Optional: scaffold clustering
    scaffolds = extract_scaffolds(mols_1)
    clusters = cluster_scaffolds(sim_matrix, n_clusters=5)
    for cluster_id in range(max(clusters) + 1):
        cluster_group = [scaffolds[i] for i in range(len(scaffolds)) if clusters[i] == cluster_id]
        img= Draw.MolsToGridImage(cluster_group, molsPerRow=3, subImgSize=(200, 200))
        img.save(os.path.join(output_path, f"scaffold_cluster_{cluster_id}.png"))
    '''

    #Optional: If two groups are not needed, comment out the next blocks
    
    # === Load group 2 ===
    smiles_2 = load_smiles(smiles_file_2)
    names_2 = load_drug_names(names_file_2)
    mols_2 = smiles_to_molecules(smiles_2)
    fps_2 = compute_fingerprints(mols_2)
    
    # === Single-group comparison for group 2 ==
    sim_matrix = compute_similarity_matrix(fps_2)
    plot_heatmap(sim_matrix, names_2, output_path, filename="similarity_heatmap_5Day.png", title="Tanimoto Similarity Heatmap - 5 Day")

    # Optional: t-SNE just for group 2
    plot_tsne_with_similarity(fps_2, names_2, output_path, filename="tsne_5Day.png")

    # === Cross-group comparison ===
    cross_sim_matrix = compute_cross_similarity_matrix(fps_1, fps_2)
    plot_heatmap(
        cross_sim_matrix,
        row_labels=names_1,
        col_labels=names_2,
        output_path=output_path,
        filename="cross_similarity_heatmap.png",
        title="Tanimoto Similarity Matrix"
    )
    # Optional: t-SNE of both sets
    combined_fps = fps_1 + fps_2
    combined_names = names_1 + names_2
    plot_tsne_with_similarity(combined_fps, combined_names, output_path, filename="tsne_combined.png")

    #Optional: End commented block for two groups  
    

if __name__ == "__main__":
    main()
