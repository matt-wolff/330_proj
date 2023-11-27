import argparse
import pandas as pd
from tqdm import tqdm
import ast
import random
from model import ProteinEmbedder
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from embeddings import ESMEmbedder
from main import defaultHypers
import os

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)


def train_each_task(args, hp, df, split_type):
    query_accs = []
    for index, row in tqdm(df.iterrows(), desc=f'Training model for each {split_type} task', total=len(df.index)):
        protein_embedder = ProteinEmbedder(hp, 'data/residues.json').to(args.device)
        protein_embedder.emb = ESMEmbedder(args.device).to(args.device)
        model = nn.Sequential(
            protein_embedder,
            nn.ReLU(),
            nn.Linear(hp["projection_space_dims"], 2)
        ).to(args.device)
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
        pos_ids = random.sample(ast.literal_eval(row["Positive IDs"]), k=8)
        neg_ids = ast.literal_eval(row["Negative IDs"])
        neg_ids = random.sample(neg_ids, k=len(neg_ids) if len(neg_ids) < len(pos_ids) else 8)
        if len(neg_ids) < len(pos_ids):
            all_other = ast.literal_eval(row["All Other IDs"])
            neg_ids += random.sample(all_other, k=8 - len(neg_ids))

        support_pos = [(0, pos_id) for pos_id in pos_ids[:-3]]
        support_neg = [(1, neg_id) for neg_id in neg_ids[:-3]]
        support_set = support_pos + support_neg
        random.shuffle(support_set)

        for epoch in range(args.epochs):
            optimizer.zero_grad()
            targets, ids = zip(*support_set)
            logits = model(list(ids))
            loss = F.cross_entropy(logits, torch.Tensor(list(targets)).long().to(args.device))
            loss.backward()
            optimizer.step()

        query_pos = [(0, pos_id) for pos_id in pos_ids[-3:]]
        query_neg = [(1, neg_id) for neg_id in neg_ids[-3:]]
        query_set = query_pos + query_neg
        random.shuffle(query_set)
        query_ids = [protein_id for (label, protein_id) in query_set]
        query_targets = [label for (label, protein_id) in query_set]

        logits = model(query_ids)
        preds = torch.argmax(logits, dim=1).to(args.device)
        query_acc = torch.sum(preds == torch.Tensor(query_targets)).to(args.device) / len(preds)
        query_accs.append(query_acc)
        torch.save(model.state_dict(), f'models/baseline/{split_type}_models/{split_type}_baseline_task_{index}.pt')
        break

    avg_acc_str = f"Average accuracy on {split_type} tasks' query sets: {torch.mean(torch.Tensor(query_accs))}"
    print(avg_acc_str)
    with open(f'models/baseline/{split_type}_models/{split_type}_baseline_avg_acc.txt', 'w') as f:
        f.write(avg_acc_str)


def main(args):
    os.makedirs('models', exist_ok=True)
    os.makedirs('models/baseline', exist_ok=True)
    os.makedirs('models/baseline/test_models', exist_ok=True)
    os.makedirs('models/baseline/val_models', exist_ok=True)

    val_df = pd.read_csv('data/val_go_tasks.csv', encoding='utf-8')
    test_df = pd.read_csv('data/test_go_tasks.csv', encoding='utf-8')
    hp = defaultHypers(args.learning_rate)
    train_each_task(args, hp, val_df, 'val')
    train_each_task(args, hp, test_df, 'test')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train baseline models.')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                        help='learning rate for the network')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()
    main(args)
