from Bio.PDB import *
from Bio.Data.IUPACData import protein_letters_3to1
import os
DATA_PATH = 'data/UP000005640_9606_HUMAN_v4'

parser = PDBParser()
for filename in os.listdir('./data/UP000005640_9606_HUMAN_v4'):
    if filename.endswith(".pdb"):
        structure = parser.get_structure('foobar', f'{DATA_PATH}/{filename}')
        all_residuals = ""
        chain_residuals = []
        for chain in structure.get_chains():
            res_list = Selection.unfold_entities(chain, "R")
            chain_residual = ""
            for res in res_list:
                chain_residual += protein_letters_3to1[res.get_resname().capitalize()]
            chain_residuals.append(chain_residual)
            all_residuals += chain_residual

