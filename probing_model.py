import argparse
import sys
sys.path.append('..')

import pandas as pd
import torch.optim as optim
from model import ballClassifier,ProteinEmbedder, PEwoPostCombination,PEwoDirectEmbedding,PEwoGAT
import ast
import random
import torch.nn.functional as F
from torch.utils import tensorboard
import torch_geometric as pyg
import torch
from embeddings import ESMEmbedder
from tqdm import tqdm
import copy

from main import get_support_and_query_ids

hyper = dict()
hyper["num_epochs"] = 5
hyper["inmodel_type"] = ProteinEmbedder
hyper["ball_radius"] = 1
hyper["projection_space_dims"] = 32
hyper["gat_layers"] = 2
hyper["gat_hidden_size"] = 512
hyper["gat_dropout"] = 0.0
hyper["postcomb_dim"] = 512
hyper["postcomb_layers"] = 4

DEVICE = "cuda"

model = ballClassifier(hyper, batchSize=1)
emb = ESMEmbedder(DEVICE).to(DEVICE)
model.model.emb = emb

model.load_state_dict(torch.load('ball_run5_epoch4.pt')) # CHANGE WITH MODEL YOU WANT TO PROBE
model.to(DEVICE)

# Computing Test accuracy
train_df = pd.read_csv('data/train_go_tasks.csv', encoding='utf-8')

net_accuracy = []

for index, row in tqdm(train_df.iterrows(), desc=f'Computing Test Accuracy', total=len(train_df.index)):  # Iterating over each task
    support_ids, query_pos_ids, query_neg_ids = get_support_and_query_ids(row)
    probs = model(support_ids, query_pos_ids + query_neg_ids)
    targets = torch.Tensor([0,0,0,1,1,1]).to(DEVICE).to(torch.int64)
    accuracy = torch.mean((torch.argmax(probs,dim=1)==targets).float()).item()
    net_accuracy.append(accuracy)

print(torch.mean(torch.Tensor(net_accuracy)))