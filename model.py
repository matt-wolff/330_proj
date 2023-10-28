from Bio.PDB import *
DATA_PATH = 'data/UP000005640_9606_HUMAN_v4'

parser = PDBParser()
structure = parser.get_structure('PHA-L', f'{DATA_PATH}/AF-A0A0A0MRZ7-F1-model_v4.pdb')  # User-generated label, file name
all_residues = []
for model in structure:
    for chain in model:
        all_residues += ["CHAIN_START"]
        res_list = Selection.unfold_entities(chain, "R")
        res_list = [res.get_resname() for res in res_list]
        all_residues += res_list + ["CHAIN_END"]
print(all_residues)
