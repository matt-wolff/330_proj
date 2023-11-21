from Bio.UniProt.GOA import gafiterator
import pandas as pd
import json
DATA_PATH = 'data'

with open(f'{DATA_PATH}/residues.json', 'r') as f:
    ID_TO_RESIDUES = json.load(f)

function_to_protein = {}
all_proteins = set()
with open(f'{DATA_PATH}/goa_human.gaf', 'r') as handle:
    for rec in gafiterator(handle):
        if rec['Aspect'] == 'F' and rec['DB'] == 'UniProtKB' and 'enables' in rec["Qualifier"] and rec['DB_Object_ID'] in ID_TO_RESIDUES.keys():
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
for go_tag, go_dict in function_to_protein.items():
    pos_ids, neg_ids = go_dict['positive_ids'], go_dict['negative_ids']
    if len(pos_ids) >= 8:  # So that each task can have at least 5 pos support examples and 3 pos queries
        data["GO Functions"].append(go_tag)
        data["Positive IDs"].append(list(pos_ids))
        data["Negative IDs"].append(list(neg_ids))
        data["All Other IDs"].append(list(all_proteins - pos_ids - neg_ids))

pdb_df = pd.DataFrame(data)
pdb_df.to_csv('data/go_tasks.csv', index=False)
