import pandas as pd
import torch.optim as optim
from model import ballClassifier
import json
import ast
import random
import torch.nn.functional as F

# TRAIN_PATH = 'data/go_tasks.csv'
# VAL_PATH = 'data/go_tasks.csv'
PATH = 'data/go_tasks.csv'
with open('data/residues.json', 'r') as f:
  ID_TO_RESIDUES = json.load(f)

df = pd.read_csv(PATH, encoding='utf-8')
num_tasks, _ = df.shape # Note: _ = 3.
# I chose a random split, we can modify this
train_df = df[:int(0.6*num_tasks)]
val_df = df[int(0.6*num_tasks):int(0.8*num_tasks)]
test_df = df[int(0.8*num_tasks):]
learning_rates = [1e-6, 5e-6, 1e-5]
num_epochs = 5 # Should we modify the number of epochs?

for lr in learning_rates:
    ball = ballClassifier(batchSize=8, jsonSeqFile='data/residues.json')
    optimizer = optim.AdamW(ball.parameters(), lr=lr)
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
            probs = ball(support_ids, query_pos_ids + query_neg_ids)
            targets = torch.Tensor([[1,0],[1,0],[1,0],[0,1],[0,1],[0,1]])
            loss = F.cross_entropy(torch.log(probs), labels)
            
            loss.backward()
            optimizer.step()