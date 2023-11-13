from torch.utils.data import DataLoader
import pandas as pd
import torch.optim as optim
from model import ProteinEmbedder, knnClassifier
import tqdm as tqdm

TRAIN_PATH = 'data/residues.csv'
VAL_PATH = 'data/residues.csv'

train_df = pd.read_csv(TRAIN_PATH, encoding='utf-8')
val_df = pd.read_csv(VAL_PATH, encoding='utf-8')
train_dataset = ProteinDataset(train_df)
val_dataset = ProteinDataset(val_df)
learning_rates = [1e-6, 5e-6, 1e-5]
batch_sizes = [16, 32, 64]
num_epochs = 5

for batch_size in batch_sizes:
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    for lr in learning_rates:
        embedder = ProteinEmbedder()
        knn = knnClassifier(batch_size=16, k=2)  # output is [numExamples, 2]
        optimizer = optim.AdamW(embedder.parameters(), lr=lr)
        train_losses, val_losses = [], []
        for epoch in range(num_epochs):
            train_loss = 0
            for batch in tqdm(train_dataloader, desc=f'Training epoch: {epoch}'):
                optimizer.zero_grad()
