from Bio.PDB import *
from Bio.Data.IUPACData import protein_letters_3to1
from Bio.UniProt.GOA import gafiterator
import os
import pandas as pd
DATA_PATH = 'data/UP000005640_9606_HUMAN_v4'

parser = PDBParser()
data = {
    "UnitProtKB accession number": [],
    "Residues": []
}
for filename in os.listdir('./data/UP000005640_9606_HUMAN_v4'):
    if filename.endswith(".pdb"):
        structure = parser.get_structure('foobar', f'{DATA_PATH}/{filename}')
        code_start = structure.header["name"].find("(") + 1
        code_end = structure.header["name"].find(")")
        uniprot = structure.header["name"][code_start:code_end]  # UniProtKB accession number
        data['UnitProtKB accession number'].append(uniprot)

        all_residues = ""
        for chain in structure.get_chains():
            res_list = Selection.unfold_entities(chain, "R")
            for res in res_list:
                all_residues += protein_letters_3to1[res.get_resname().capitalize()]
        data['Residues'].append(all_residues)

pdb_df = pd.DataFrame(data)
pdb_df.to_csv('data/residues.csv', index=False)
