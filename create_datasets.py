from Bio.PDB import *
from Bio.Data.IUPACData import protein_letters_3to1
import os
import matplotlib.pyplot as plt
from Bio.UniProt.GOA import gafiterator
import pandas as pd
import json
from tqdm import tqdm

DATA_PATH = 'data'
SEED = 42

# Parse PDB files to create an ID to residues dictionary
pdb_parser = PDBParser()
id_to_residues = {}
lengths = []
for filename in tqdm(os.listdir(f'{DATA_PATH}/UP000005640_9606_HUMAN_v4')):
    if filename.endswith(".pdb"):
        uniprot = filename.split('-')[1]  # UniProtKB Object ID
        pdb_structure = pdb_parser.get_structure('foobar', f'{DATA_PATH}/UP000005640_9606_HUMAN_v4/{filename}')
        all_residues = ""
        for res in pdb_structure.get_residues():
            all_residues += protein_letters_3to1[res.get_resname().capitalize()]
        if len(all_residues) <= 1000:
            id_to_residues[uniprot] = all_residues
        lengths.append(len(all_residues))
plt.hist(lengths)
plt.savefig('num_residues_chart.png')

with open(f"{DATA_PATH}/residues.json", "w") as f:
    json.dump(id_to_residues, f)

# Parse goa_human.gaf to create a df with GO tags and their associated positive and negative IDs
function_to_protein = {}
all_proteins = set()
with open(f'{DATA_PATH}/goa_human.gaf', 'r') as handle:
    for rec in tqdm(gafiterator(handle)):
        if rec['Aspect'] == 'F' and rec['DB'] == 'UniProtKB' and 'enables' in rec["Qualifier"] and rec['DB_Object_ID'] in id_to_residues.keys():
            if rec['GO_ID'] not in function_to_protein.keys():
                function_to_protein[rec['GO_ID']] = {}
                function_to_protein[rec['GO_ID']]['positive_ids'] = set()
                function_to_protein[rec['GO_ID']]['negative_ids'] = set()

            if 'NOT' in rec['Qualifier']:
                function_to_protein[rec['GO_ID']]['negative_ids'].add(rec['DB_Object_ID'])
            else:
                function_to_protein[rec['GO_ID']]['positive_ids'].add(rec['DB_Object_ID'])
            all_proteins.add(rec['DB_Object_ID'])

data = {
    "GO Functions": [],
    "Positive IDs": [],
    "Negative IDs": [],
    "All Other IDs": []
}
for go_tag, go_dict in tqdm(function_to_protein.items()):
    pos_ids, neg_ids = go_dict['positive_ids'], go_dict['negative_ids']
    if len(pos_ids) >= 8:  # So that each task can have at least 5 pos support examples and 3 pos queries
        data["GO Functions"].append(go_tag)
        data["Positive IDs"].append(list(pos_ids))
        data["Negative IDs"].append(list(neg_ids))
        data["All Other IDs"].append(list(all_proteins - pos_ids - neg_ids))

df = pd.DataFrame(data)

# Split dataframe into train, val, and test sets
num_tasks, _ = df.shape  # Note: _ = 3.
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
train_df = df[:int(0.7*num_tasks)]
train_df.to_csv('data/train_go_tasks.csv', index=False)
val_df = df[int(0.7*num_tasks):int(0.8*num_tasks)]
val_df.to_csv('data/val_go_tasks.csv', index=False)
test_df = df[int(0.8*num_tasks):]
test_df.to_csv('data/test_go_tasks.csv', index=False)
