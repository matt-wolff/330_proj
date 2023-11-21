import argparse
import os
import sys
sys.path.append('..')

import pandas as pd
import torch.optim as optim
from model import ballClassifier
import json
import ast
import random
import torch.nn.functional as F
from torch.utils import tensorboard
import torch_geometric as pyg
import torch
from embeddings import ESMEmbedder
from tqdm import tqdm

VAL_INTERVAL = 50
DEVICE = 'cuda'
# TRAIN_PATH = 'data/go_tasks.csv'
# VAL_PATH = 'data/go_tasks.csv'
PATH = 'data/go_tasks.csv'
# learning_rates = [1e-6, 5e-6, 1e-5]

def main(args):

    if args.device == "cuda" and torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"

    log_dir = f'./logs/milestone_run_1'
    print(f'log_dir: {log_dir}')
    writer = tensorboard.SummaryWriter(log_dir=log_dir)

    with open('data/residues.json', 'r') as f:
        ID_TO_RESIDUES = json.load(f)

    df = pd.read_csv(PATH, encoding='utf-8')
    num_tasks, _ = df.shape # Note: _ = 3.
    # I chose a random split, we can modify this
    df = df.sample(frac=1).reset_index(drop=True)
    train_df = df[:int(0.7*num_tasks)]
    val_df = df[int(0.7*num_tasks):int(0.8*num_tasks)]
    test_df = df[int(0.8*num_tasks):]
    num_epochs = 5 # Should we modify the number of epochs?

    lr = args.learning_rate

    ball = ballClassifier(batchSize=8, jsonSeqFile='data/residues.json').to(DEVICE)

    def initializeParams(module): ## TODO when you add in the pretrained model; ensure you do not initialize that
        if isinstance(module, torch.nn.Linear):
            module.weight.data = torch.nn.init.xavier_normal_(module.weight.data, gain=torch.nn.init.calculate_gain('relu'))
            if module.bias is not None:
                module.bias.data.zero_()

        if isinstance(module, torch.nn.LayerNorm) or isinstance(module, torch.nn.BatchNorm1d) or isinstance(module, torch.nn.BatchNorm2d):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
        if isinstance(module, pyg.nn.models.GAT):
            module.reset_parameters()

    ball.apply(initializeParams)
    
    emb = ESMEmbedder(DEVICE).to(DEVICE)
    ball.model.emb = emb

    optimizer = optim.AdamW(ball.parameters(), lr=lr)
    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        train_loss = 0
        for index, row in tqdm(train_df.iterrows(), desc=f'Training epoch: {epoch}'):  # Iterating over each task
            optimizer.zero_grad()

            pos_ids = ast.literal_eval(row["Positive IDs"])
            rand_pos_ids = random.sample(pos_ids, k=8)
            support_ids = rand_pos_ids[:5]
            query_pos_ids = rand_pos_ids[5:]

            neg_ids = ast.literal_eval(row["Negative IDs"])
            if len(neg_ids) < 3:
                all_other = ast.literal_eval(row["All Other IDs"])
                query_neg_ids = random.sample(neg_ids, k=len(neg_ids))
                query_neg_ids += random.sample(all_other, k=3-len(neg_ids))
            else:
                query_neg_ids = random.sample(neg_ids, k=3)

            probs = ball(support_ids, query_pos_ids + query_neg_ids)
            # targets = torch.Tensor([[0,1],[0,1],[0,1],[1,0],[1,0],[1,0]])
            targets = torch.Tensor([1,1,1,0,0,0]).to(DEVICE).to(torch.int64)
            loss = F.cross_entropy(torch.log(probs), targets)

            loss.backward()
            optimizer.step()

            i_step = num_tasks*epoch + index
            writer.add_scalar('loss/train', loss.item(), i_step)
            accuracy = torch.mean((torch.argmax(probs,dim=1)==targets).float()).item()
            writer.add_scalar(
                'train_accuracy/',
                accuracy,
                i_step
            )

    torch.save(ball.state_dict(), 'ball.pt')

            # if index % VAL_INTERVAL == 0:
            #     print("Start Validation...")
            #     with torch.no_grad():
            #         losses, accuracies = [], []
            #         for iter_val, row_val in tqdm(val_df.iterrows(), desc=f"Val Training Epoch {epoch}"):
            #             print(torch.cuda.memory_summary())
            #             pos_ids_val, neg_ids_val = ast.literal_eval(row_val["Positive IDs"]), ast.literal_eval(row_val["Negative IDs"])
            #             rand_pos_ids_val = random.sample(pos_ids_val, k=8)
            #             support_ids_val = rand_pos_ids_val[:5]
            #             query_pos_ids_val = rand_pos_ids_val[5:]
            #             query_neg_ids_val = random.sample(neg_ids_val, k=3)
            #             probs = ball(support_ids_val, query_pos_ids_val + query_neg_ids_val)
            #             targets = torch.Tensor([1,1,1,0,0,0]).to(DEVICE).to(torch.int64)
            #             loss = F.cross_entropy(torch.log(probs), targets)
            #             accuracy = torch.mean((torch.argmax(probs,dim=1)==targets).float()).item()
            #             losses.append(loss)
            #             accuracies.append(accuracy)
            #         loss_val = np.mean(losses)
            #         accuracies_val = np.mean(accuracies)

            #         writer.add_scalar('loss/val', loss_val, i_step)
            #         writer.add_scalar(
            #             'val_accuracy',
            #             accuracy_val,
            #             i_step
            #         )

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train a Ball Classifier!')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                        help='learning rate for the network')
    parser.add_argument('--device', type=str, default='cuda')
    
    # parser.add_argument('--datadir', default='xxx', type=str, help='directory for datasets.')

    args = parser.parse_args()

    main(args)