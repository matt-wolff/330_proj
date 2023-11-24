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
import copy

VAL_INTERVAL = 50
DEVICE = 'cuda'
# TRAIN_PATH = 'data/go_tasks.csv'
# VAL_PATH = 'data/go_tasks.csv'
PATH = 'data/go_tasks.csv'
# learning_rates = [1e-6, 5e-6, 1e-5]
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)


def get_support_and_query_ids(row):
    pos_ids = ast.literal_eval(row["Positive IDs"])
    rand_pos_ids = random.sample(pos_ids, k=8)
    support_ids = rand_pos_ids[:5]
    query_pos_ids = rand_pos_ids[5:]

    neg_ids = ast.literal_eval(row["Negative IDs"])
    if len(neg_ids) < 3:
        all_other = ast.literal_eval(row["All Other IDs"])
        query_neg_ids = random.sample(neg_ids, k=len(neg_ids))
        query_neg_ids += random.sample(all_other, k=3 - len(neg_ids))
    else:
        query_neg_ids = random.sample(neg_ids, k=3)

    return support_ids, query_pos_ids, query_neg_ids

def getRangeCombos(space):
    hypers = list()
    hypers.append(dict())
    for hname,hrange in space.items():
        nhypers = list()
        for hyper in hypers:
            for hval in hrange:
                nhyper = copy.deepcopy(hyper)
                nhyper[hname] = hval
                nhypers.append(nhyper)
        hypers = nhypers
    return hypers
            

def main(args):

    if args.device == "cuda" and torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"


    #with open('data/residues.json', 'r') as f:
    #    ID_TO_RESIDUES = json.load(f)

    df = pd.read_csv(PATH, encoding='utf-8')
    num_tasks, _ = df.shape # Note: _ = 3.
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    train_df = df[:int(0.7*num_tasks)]
    val_df = df[int(0.7*num_tasks):int(0.8*num_tasks)]
    test_df = df[int(0.8*num_tasks):]

    if args.mode=="train":
        hyper=dict()
        hyper["learning_rate"] = args.learning_rate
        hyper["num_epochs"] = 5
        hyper["run_name"] = args.run_name
        train(hyper,train_df,val_df,DEVICE)
    elif args.mode=="hp_search":
        hyperranges=dict()
        hyperranges["learning_rate"] = [1e-4,3e-4,1e-3]
        hyperranges["num_epochs"] = [5,10]
        hyperranges["run_name"] = [args.run_name] # I think this is a bad idea with hps NOTE 
        hypers = getRangeCombos(hyperranges)
        import pdb
        pdb.set_trace()
        for hyper in hypers:
            train(hyper,train_df,val_df,DEVICE)

        

def train(hyper,train_df,val_df,DEVICE):
    log_dir = f'./logs/{hyper["run_name"]}'
    print(f'log_dir: {log_dir}')
    writer = tensorboard.SummaryWriter(log_dir=log_dir)

    lr = hyper["learning_rate"]
    num_epochs = hyper["num_epochs"]
    ball = ballClassifier(batchSize=8).to(DEVICE)
    num_train_tasks, _ = train_df.shape

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
    for epoch in range(num_epochs):
        for index, row in tqdm(train_df.iterrows(), desc=f'Training epoch: {epoch}'):  # Iterating over each task
            optimizer.zero_grad()

            support_ids, query_pos_ids, query_neg_ids = get_support_and_query_ids(row)
            _, queryDists = ball.get_prototype_and_query_dists(support_ids, query_pos_ids + query_neg_ids)

            # contrastive loss https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1640964
            pos_dists = queryDists[:3]
            neg_dists = queryDists[3:]
            loss = 0.5 * torch.sum(torch.pow(pos_dists, 2))
            loss += 0.5 * torch.sum(torch.pow(torch.max(torch.Tensor([0]).to(DEVICE), ball.radius - neg_dists), 2))  # Doesn't add to loss if dist >= margin

            loss.backward()
            optimizer.step()

            i_step = num_train_tasks*epoch + index
            writer.add_scalar('loss/train', loss.item(), i_step)

            if index % VAL_INTERVAL == 0:
                print("Start Validation...")
                with torch.no_grad():
                    val_losses, val_accuracies = [], []
                    for iter_val, row_val in tqdm(val_df.iterrows(), desc=f"Val Training Epoch {epoch}"):
                        support_ids_val, query_pos_ids_val, query_neg_ids_val = get_support_and_query_ids(row_val)
                        probs = ball(support_ids_val, query_pos_ids_val + query_neg_ids_val)
                        targets = torch.Tensor([1,1,1,0,0,0]).to(DEVICE).to(torch.int64)
                        loss = F.nll_loss(torch.log(probs), targets)
                        accuracy = torch.mean((torch.argmax(probs,dim=1)==targets).float()).item()
                        val_losses.append(loss.item())
                        val_accuracies.append(accuracy)
                        break
                    loss_val = torch.mean(torch.Tensor(val_losses))
                    accuracy_val = torch.mean(torch.Tensor(val_accuracies))

                    writer.add_scalar('loss/val', loss_val, i_step)
                    writer.add_scalar(
                        'val_accuracy',
                        accuracy_val,
                        i_step
                    )

                if (index + VAL_INTERVAL) % num_train_tasks > index % num_train_tasks:
                    torch.save(ball.state_dict(), f'ball_run2_epoch{epoch}.pt')


if __name__ == '__main__':
    modes = ['train',"hp_search"]

    parser = argparse.ArgumentParser('Train a Ball Classifier!')
    parser.add_argument('mode', type=str, help=(' '.join(modes)))
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                        help='learning rate for the network')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--run_name', type=str, default='temp')

    args = parser.parse_args()
    assert(args.mode in modes)

    main(args)
