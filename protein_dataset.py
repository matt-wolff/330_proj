from torch.utils.data import Dataset


class ProteinDataset(Dataset):
    def __init__(self, df):
        self.input_ids = df['input_ids']

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return (self.input_ids[idx],)
