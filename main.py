from torch.utils.data import DataLoader
import torch.optim as optim
from model import ProteinEmbedder

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
        optimizer = optim.AdamW(embedder.parameters(), lr=lr)
        train_losses, val_losses = [], []
