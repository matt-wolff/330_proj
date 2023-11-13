import pandas as pd
import torch.optim as optim
from model import ProteinEmbedder, knnClassifier
import json
import ast
import random

TRAIN_PATH = 'data/go_tasks.csv'
VAL_PATH = 'data/go_tasks.csv'
with open('data/residues.json', 'r') as f:
  ID_TO_RESIDUES = json.load(f)

train_df = pd.read_csv(TRAIN_PATH, encoding='utf-8')
val_df = pd.read_csv(VAL_PATH, encoding='utf-8')
learning_rates = [1e-6, 5e-6, 1e-5]
num_epochs = 5

for lr in learning_rates:
    embedder = ProteinEmbedder()
    # knn = knnClassifier(batchSize=16)  # output is [numExamples, 2]
    optimizer = optim.AdamW(embedder.parameters(), lr=lr)
    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        train_loss = 0
        for index, row in train_df.iterrows():  # Iterating over each task
            optimizer.zero_grad()

            pos_ids, neg_ids = ast.literal_eval(row["Positive IDs"]), ast.literal_eval(row["Negative IDs"])
            rand_pos_ids = random.sample(pos_ids, k=8)
            support_ids = rand_pos_ids[:5]
            query_pos_ids = rand_pos_ids[5:]
            query_neg_ids = random.sample(neg_ids, k=3)

            support_residues = [ID_TO_RESIDUES[protein_id] for protein_id in support_ids]
            query_pos_residues = [ID_TO_RESIDUES[protein_id] for protein_id in query_pos_ids]
            query_neg_residues = [ID_TO_RESIDUES[protein_id] for protein_id in query_neg_ids]

            embeddings = embedder(support_residues + query_pos_residues + query_neg_residues,
                                  support_ids + query_pos_ids + query_pos_ids)
            support_embeddings = embeddings[:len(support_residues)]
            query_pos_embeddings = embeddings[len(support_residues):len(support_residues)+len(pos_ids)]
            query_neg_embeddings = embeddings[len(support_residues)+len(pos_ids):]

            # Step after each task



