import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Draw
from rdkit.Chem import rdFMCS  # Import the MCS module
from sklearn.manifold import TSNE
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.cluster import AgglomerativeClustering
import os  # Added for handling file paths

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
    fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits) for mol in molecules]
    return fingerprints

# Compute Tanimoto similarity matrix
def compute_similarity_matrix(fingerprints):
    num_mols = len(fingerprints)
    similarity_matrix = np.zeros((num_mols, num_mols))
    for i in range(num_mols):
        for j in range(num_mols):
            similarity_matrix[i, j] = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
    return similarity_matrix

# Extract common scaffolds
def extract_scaffolds(molecules):
    scaffolds = [MurckoScaffold.GetScaffoldForMol(mol) for mol in molecules]
    return scaffolds

# Find Maximum Common Substructure (MCS)
def find_mcs(molecules):
    # Perform MCS search on the molecules
    mcs_result = rdFMCS.FindMCS(molecules)

    # Check if an MCS was found
    if mcs_result.canceled:
        print("MCS search was canceled.")
        return None  # No common substructure found
    
    if mcs_result.numAtoms == 0:
        print("No common substructure found.")
        return None  # No common substructure found
    # Get the MCS molecule
    mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
    return mcs_mol

# Highlight the MCS on the molecules
def highlight_mcs(molecules, mcs_mol):
    # Generate a list of molecule images with the MCS highlighted
    highlighted_mols = []
    for mol in molecules:
        match = mol.GetSubstructMatch(mcs_mol)  # Find the MCS in the molecule
        img = Draw.MolToImage(mol, highlightAtoms=match)  # Highlight the atoms in the MCS
        highlighted_mols.append(img)
    return highlighted_mols

# Cluster scaffolds based on similarity
def cluster_scaffolds(similarity_matrix, n_clusters=5):
    # Convert similarity to dissimilarity (distance matrix)
    distance_matrix = 1 - similarity_matrix
    
    # Perform agglomerative clustering
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='average')
    clusters = clustering.fit_predict(distance_matrix)  # Use the distance matrix here
    
    return clusters

# Plot the heatmap
def plot_heatmap(similarity_matrix, drug_names_list, output_path):
    plt.figure(figsize=(8, 6))  # Slightly increase the figure size
    sns.heatmap(
        similarity_matrix,
        xticklabels=drug_names_list,
        yticklabels=drug_names_list,
        annot=True,
        fmt=".2f",  # Format the numbers to 2 decimal places
        cmap='coolwarm',
        annot_kws={"size": 5.7},  # Reduce the annotation font size by 5% for better readability
        cbar_kws={"label": "Tanimoto Similarity"},
        linewidths=0.5,  # Add some separation between cells
        linecolor='black',  # Color of lines between cells
        square=True  # Make the heatmap square for better readability
    )
    
    # Adjust tick labels to use drug names instead of SMILES
    plt.xticks(ticks=np.arange(len(drug_names_list)) + 0.5, labels=drug_names_list, rotation=45, ha="right", fontsize=7.6)
    plt.yticks(ticks=np.arange(len(drug_names_list)) + 0.5, labels=drug_names_list, rotation=0, fontsize=7.6)
    
    plt.title("Tanimoto Similarity Heatmap", fontsize=16)
    plt.tight_layout(pad=2.0)  # Adjust layout to prevent clipping of labels
    plt.savefig(os.path.join(output_path, "similarity_heatmap.png"))  # Save heatmap
    plt.close()

# Apply t-SNE for visualization
def plot_tsne(fingerprints, drug_names_list, output_path):
    tsne = TSNE(n_components=2, perplexity=2, random_state=42)
    
    # Convert fingerprints to numpy array
    fingerprint_array = np.array([list(fp) for fp in fingerprints])
    
    embeddings = tsne.fit_transform(fingerprint_array)
    plt.figure(figsize=(6, 5))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c='blue')
    for i, txt in enumerate(drug_names_list):
        plt.annotate(txt, (embeddings[i, 0], embeddings[i, 1]), fontsize=6)
    plt.title("t-SNE Visualization of Molecular Similarity")
    plt.savefig(os.path.join(output_path, "tsne_plot.png"))  # Save t-SNE plot
    plt.close()

# Visualize common scaffolds
def plot_scaffolds(scaffolds, output_path):
    img = Draw.MolsToGridImage(scaffolds, molsPerRow=3, subImgSize=(200, 200))
    img.save(os.path.join(output_path, "scaffold_grid.png"))  # Save scaffold grid

# Main function
def main():
    smiles_file_path = "/Users/jakewood/Downloads/UTCOMLS/UTCOMLS Research/Dr. Robert Smith/KimSin/JW_Sim_Calc/test_SMILES.txt"  # SMILES file
    drug_names_file_path = "/Users/jakewood/Downloads/UTCOMLS/UTCOMLS Research/Dr. Robert Smith/KimSin/JW_Sim_Calc/test_drug_names.txt"  # File containing drug names
    
    output_path = "/Users/jakewood/Downloads/UTCOMLS/UTCOMLS Research/Dr. Robert Smith/KimSin/JW_Sim_Calc/Test_OutPut"  # Directory where images will be saved
    os.makedirs(output_path, exist_ok=True)  # Create directory if it doesn't exist
    
    # Load SMILES and drug names
    smiles_list = load_smiles(smiles_file_path)
    drug_names_list = load_drug_names(drug_names_file_path)
    
    molecules = smiles_to_molecules(smiles_list)
    fingerprints = compute_fingerprints(molecules)
    similarity_matrix = compute_similarity_matrix(fingerprints)
    scaffolds = extract_scaffolds(molecules)
    
    # Find the MCS
    mcs_mol = find_mcs(molecules)
    
    # Highlight the MCS on the molecules
    highlighted_mols = highlight_mcs(molecules, mcs_mol)
    
    # Save highlighted molecules with MCS
    for idx, img in enumerate(highlighted_mols):
        img.save(os.path.join(output_path, f"highlighted_mcs_mol_{idx}.png"))

    # Cluster scaffolds based on similarity
    clusters = cluster_scaffolds(similarity_matrix, n_clusters=5)

    # Use clustering results to create a grouped scaffold grid
    cluster_scaffolds_grouped = []
    for cluster_id in range(max(clusters) + 1):
        cluster_scaffolds_grouped.append([scaffolds[i] for i in range(len(scaffolds)) if clusters[i] == cluster_id])
    
    # Pass drug names to plot function
    plot_heatmap(similarity_matrix, drug_names_list, output_path)
    plot_tsne(fingerprints, drug_names_list, output_path)
    plot_scaffolds(scaffolds, output_path)
    
    # Optionally save the grouped scaffold grid
    for idx, cluster in enumerate(cluster_scaffolds_grouped):
        img = Draw.MolsToGridImage(cluster, molsPerRow=3, subImgSize=(200, 200))
        img.save(os.path.join(output_path, f"scaffold_grid_cluster_{idx}.png"))

if __name__ == "__main__":
    main()
