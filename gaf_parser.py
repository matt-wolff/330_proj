from Bio.UniProt.GOA import gafiterator
import pandas as pd
DATA_PATH = 'data'

data = {
    "GO Functions": [],
    "Positive IDs": [],
    "Negative IDs": []
}

functions_to_id = {}
all_proteins = set()
with open(f'{DATA_PATH}/goa_human.gaf', 'r') as handle:
    for rec in gafiterator(handle):
        if rec['Aspect'] == 'F' and rec['DB'] == 'UniProtKB' and 'NOT' not in rec["Qualifier"]:
            if rec['GO_ID'] in functions_to_id.keys():
                functions_to_id[rec['GO_ID']].add(rec['DB_Object_ID'])
            else:
                functions_to_id[rec['GO_ID']] = {rec['DB_Object_ID']}
            all_proteins.add(rec['DB_Object_ID'])

for go_func, pos_ids in functions_to_id.items():
    if len(pos_ids) >= 8:  # So that each task can have at least 5 pos support examples and 3 pos queries
        data["GO Functions"].append(go_func)
        data["Positive IDs"].append(list(pos_ids))
        data["Negative IDs"].append(list(all_proteins - pos_ids))


pdb_df = pd.DataFrame(data)
pdb_df.to_csv('data/go_tasks.csv', index=False)
