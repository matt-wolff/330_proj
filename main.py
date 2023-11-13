import pandas as pd
import torch.optim as optim
from model import ballClassifier
import json
import ast
import random
import torch.nn.functional as F
from torch.utils import tensorboard

VAL_INTERVAL = 20

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
            loss = F.cross_entropy(torch.log(probs), labels)
            
            loss.backward()
            optimizer.step()

            writer.add_scalar('loss/train', loss.item(), i_step)
            accuracy_query = torch.mean((torch.argmax(probs,dim=1)==targets).float()).item()
            writer.add_scalar(
                'train_accuracy/query',
                accuracy_query.item(),
                i_step
            )

            # if i_step % self.val_interval == 0:
            #     print("Start Validation...")
            #     with torch.no_grad():
            #         losses, accuracies = [], [], []
            #         for i, val_task_batch in enumerate(dataloader_meta_val):
            #             if self.bio and i > 600:
            #                 break
            #             loss, accuracy_support, accuracy_query = (
            #                 self._step(val_task_batch)
            #             )
            #             losses.append(loss.item())
            #             accuracies_support.append(accuracy_support)
            #             accuracies_query.append(accuracy_query)
            #         loss = np.mean(losses)
            #         accuracy_support = np.mean(accuracies_support)
            #         accuracy_query = np.mean(accuracies_query)
            #         ci95 = 1.96 * np.std(accuracies_query) / np.sqrt(600 * 4)
            #     if self.bio:
            #         print(
            #             f'Validation: '
            #             f'loss: {loss:.3f}, '
            #             f'support accuracy: {accuracy_support:.3f}, '
            #             f'query accuracy: {accuracy_query:.3f}',
            #             f'Ci95: {ci95:.3f}'
            #         )
            #     else: 
            #         print(
            #             f'Validation: '
            #             f'loss: {loss:.3f}, '
            #             f'support accuracy: {accuracy_support:.3f}, '
            #             f'query accuracy: {accuracy_query:.3f}'
            #         )
            #     writer.add_scalar('loss/val', loss, i_step)
            #     writer.add_scalar(
            #         'val_accuracy/support',
            #         accuracy_support,
            #         i_step
            #     )
            #     writer.add_scalar(
            #         'val_accuracy/query',
            #         accuracy_query,
            #         i_step
            #     )
            #     if self.bio:
            #         writer.add_scalar(
            #             'val_accuracy/ci95',
            #             ci95,
            #             i_step
            #         )