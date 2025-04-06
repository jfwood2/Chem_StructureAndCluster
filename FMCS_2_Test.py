from rdkit import Chem
from rdkit.Chem import rdFMCS, AllChem
from rdkit.Chem import Draw
from rdkit import DataStructs  # Import DataStructs for Tanimoto similarity
from PIL import Image
import numpy as np

def read_smiles_from_file(file_path):
    # Read SMILES from the txt file and return them as a list
    with open(file_path, 'r') as file:
        smiles_list = file.readlines()
    
    # Remove any extra whitespace like newlines
    smiles_list = [smiles.strip() for smiles in smiles_list]
    return smiles_list

def find_mcs_and_highlight(smiles_list):
    # Convert SMILES to RDKit molecules
    molecules = [Chem.MolFromSmiles(smiles) for smiles in smiles_list if smiles]
    
    # Set up the MCS parameters
    mcs_params = rdFMCS.MCSParameters()
    mcs_params.timeout = 60
    mcs_params.minAtoms = 4
    mcs_params.matchValences = True
    mcs_params.ringMatchesRingOnly = True
    
    # Find the Maximum Common Substructure (MCS)
    mcs_result = rdFMCS.FindMCS(molecules, mcs_params)
    
    # Get the MCS structure as a molecule
    mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
    
    # Highlight the MCS in each molecule
    highlighted_mols = []
    for mol in molecules:
        # Highlight the MCS in the molecule
        highlight_atoms = mol.GetSubstructMatches(mcs_mol)
        highlight_atoms = [atom[0] for atom in highlight_atoms]  # Get atom indices of the matched substructure
        img = Draw.MolToImage(mol, highlightAtoms=highlight_atoms, size=(300, 300))
        highlighted_mols.append(img)
    
    # Show the MCS and highlighted molecules
    print("Initial MCS SMILES:", mcs_result.smartsString)
    return molecules, mcs_mol, highlighted_mols

def calculate_pairwise_similarity(molecules):
    # Calculate the pairwise similarity between the molecules based on fingerprints
    fingerprints = [AllChem.GetMorganFingerprint(mol, 2) for mol in molecules]
    
    # Compute the pairwise Tanimoto similarity
    similarity_matrix = np.zeros((len(fingerprints), len(fingerprints)))
    for i in range(len(fingerprints)):
        for j in range(i, len(fingerprints)):
            similarity = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
            similarity_matrix[i, j] = similarity_matrix[j, i] = similarity
    return similarity_matrix


def refine_mcs(molecules, initial_mcs):
    # Now refine the MCS by comparing pairwise similarity
    similarity_matrix = calculate_pairwise_similarity(molecules)
    
    # Define a threshold for the minimum similarity to refine the MCS (this could be adjusted)
    threshold = 0.8
    
    best_mcs_mol = initial_mcs
    best_similarity = 0
    
    # Iterate over possible MCS substructures
    for mol in molecules:
        try:
            # Sanitize the molecule before processing
            Chem.SanitizeMol(mol)
            
            for substructure in mol.GetSubstructMatches(best_mcs_mol):  # Correct method to find substructure matches
                # Extract a substructure from each molecule and compare similarity
                sub_mol = Chem.FragmentOnBonds(mol, list(substructure))
                
                # Sanitize the substructure before generating fingerprints
                Chem.SanitizeMol(sub_mol)
                
                # Generate fingerprint
                sub_fingerprint = AllChem.GetMorganFingerprint(sub_mol, 2)
                
                # Calculate the similarity to the current MCS
                similarity = np.mean([similarity_matrix[i, j] for i, j in enumerate(substructure)])
                
                # If the similarity is better than the best, update
                if similarity > best_similarity and similarity >= threshold:
                    best_mcs_mol = sub_mol
                    best_similarity = similarity
        except Exception as e:
            print(f"Error processing molecule: {e}")
    
    return best_mcs_mol


def combine_images(images, columns=3):
    # Combine all the images into a grid layout
    rows = (len(images) + columns - 1) // columns  # Calculate required number of rows
    widths, heights = zip(*(i.size for i in images))
    
    total_width = columns * max(widths)
    total_height = rows * max(heights)
    
    combined_image = Image.new('RGB', (total_width, total_height))

    # Paste each image into the combined image in the appropriate position
    y_offset = 0
    x_offset = 0
    for i, img in enumerate(images):
        combined_image.paste(img, (x_offset, y_offset))
        x_offset += img.width
        if (i + 1) % columns == 0:  # Move to next row after every 'columns' images
            x_offset = 0
            y_offset += img.height
    
    return combined_image

# Specify the path to your SMILES file
file_path = '/Users/jakewood/Downloads/UTCOMLS/UTCOMLS Research/Dr. Robert Smith/KimSin/JW_Sim_Calc/test_SMILES.txt'

# Read SMILES from file
smiles_list = read_smiles_from_file(file_path)

# Find MCS and highlight substructures in the molecules
molecules, initial_mcs, highlighted_images = find_mcs_and_highlight(smiles_list)

# Refine the MCS using pairwise similarity
refined_mcs = refine_mcs(molecules, initial_mcs)

# Display the refined MCS
print("Refined MCS SMILES:", Chem.MolToSmiles(refined_mcs))

# Combine all the images into a grid layout (3 columns in this case)
combined_img = combine_images(highlighted_images, columns=3)

# Display the combined image
combined_img.show()

# Optionally, save the combined image
combined_img.save("combined_mcs_image.png")
