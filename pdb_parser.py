from Bio.PDB import *
from Bio.Data.IUPACData import protein_letters_3to1
from Bio.UniProt.GOA import gafiterator
import os
import pandas as pd
DATA_PATH = 'data'

pdb_parser = PDBParser()
data = {
    "UniProtKB Object ID": [],
    "Residues": [],
    "GO Functions": []
}

uniprot_to_functions = {}
qualifiers = set()
with open(f'{DATA_PATH}/goa_human.gaf', 'r') as handle:
    for rec in gafiterator(handle):
        if rec['Aspect'] == 'F' and rec['DB'] == 'UniProtKB' and 'NOT' not in rec["Qualifier"]:
            if rec['DB_Object_ID'] in uniprot_to_functions.keys():
                uniprot_to_functions[rec['DB_Object_ID']].add(rec['GO_ID'])
            else:
                uniprot_to_functions[rec['DB_Object_ID']] = {rec['GO_ID']}

num_excluded_proteins = 0
i = 0
for filename in os.listdir(f'{DATA_PATH}/UP000005640_9606_HUMAN_v4'):
    if filename.endswith(".pdb"):
        uniprot = filename.split('-')[1]  # UniProtKB Object ID
        if uniprot in uniprot_to_functions.keys():
            data['UniProtKB Object ID'].append(uniprot)
            data['GO Functions'].append(list(uniprot_to_functions[uniprot]))

            pdb_structure = pdb_parser.get_structure('foobar', f'{DATA_PATH}/UP000005640_9606_HUMAN_v4/{filename}')
            all_residues = ""
            for res in pdb_structure.get_residues():
                all_residues += protein_letters_3to1[res.get_resname().capitalize()]
            data['Residues'].append(all_residues)
        else:
            print(f"No functions for uniprotkb: {uniprot}")
            num_excluded_proteins += 1
        i += 1
        if i == 10:
            break
print(f"Number of excluded proteins: {num_excluded_proteins}")

pdb_df = pd.DataFrame(data)
pdb_df.to_csv('data/residues.csv', index=False)
