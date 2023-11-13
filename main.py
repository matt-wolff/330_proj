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

VAL_INTERVAL = 50

log_dir = f'./logs/milestone_run_1'
print(f'log_dir: {log_dir}')
writer = tensorboard.SummaryWriter(log_dir=log_dir)

# TRAIN_PATH = 'data/go_tasks.csv'
# VAL_PATH = 'data/go_tasks.csv'
PATH = 'data/go_tasks.csv'
with open('data/residues.json', 'r') as f:
  ID_TO_RESIDUES = json.load(f)

df = pd.read_csv(PATH, encoding='utf-8')
num_tasks, _ = df.shape # Note: _ = 3.
# I chose a random split, we can modify this
train_df = df[:int(0.7*num_tasks)]
val_df = df[int(0.7*num_tasks):int(0.8*num_tasks)]
test_df = df[int(0.8*num_tasks):]
learning_rates = [1e-6, 5e-6, 1e-5]
num_epochs = 5 # Should we modify the number of epochs?

for lr in learning_rates:
    ball = ballClassifier(batchSize=8, jsonSeqFile='data/residues.json')

    import pdb
    pdb.set_trace()
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

    emb = ESMEmbedder()
    ball.model.emb = emb

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
            # targets = torch.Tensor([[0,1],[0,1],[0,1],[1,0],[1,0],[1,0]])
            targets = torch.Tensor([1,1,1,0,0,0])
            loss = F.cross_entropy(torch.log(probs), targets)
            
            loss.backward()
            optimizer.step()

            writer.add_scalar('loss/train', loss.item(), i_step)
            accuracy = torch.mean((torch.argmax(probs,dim=1)==targets).float()).item()
            writer.add_scalar(
                'train_accuracy/',
                accuracy.item(),
                i_step
            )

            if index % VAL_INTERVAL == 0:
                print("Start Validation...")
                with torch.no_grad():
                    losses, accuracies = []
                    for iter_val, row_val in val_df.iterrows():
                        pos_ids_val, neg_ids_val = ast.literal_eval(row_val["Positive IDs"]), ast.literal_eval(row_val["Negative IDs"])
                        rand_pos_ids_val = random.sample(pos_ids_val, k=8)
                        support_ids_val = rand_pos_ids_val[:5]
                        query_pos_ids_val = rand_pos_ids_val[5:]
                        query_neg_ids_val = random.sample(neg_ids_val, k=3)
                        probs = ball(support_ids_val, query_pos_ids_val + query_neg_ids_val)
                        targets = torch.Tensor([1,1,1,0,0,0])
                        loss = F.cross_entropy(torch.log(probs), targets)
                        accuracy = torch.mean((torch.argmax(probs,dim=1)==targets).float()).item()
                        losses.append(loss)
                        accuracies.append(accuracy)
                    loss_val = np.mean(losses)
                    accuracies_val = np.mean(accuracies)

                    writer.add_scalar('loss/val', loss_val, i_step)
                    writer.add_scalar(
                        'val_accuracy',
                        accuracy_val,
                        i_step
                    )
