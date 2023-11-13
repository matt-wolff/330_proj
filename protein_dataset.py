from torch.utils.data import Dataset


class ProteinDataset(Dataset):
    def __init__(self, df):
        self.unitprotids = df['UnitProtKB Object ID']
        self.residues = df['Residues']
        self.gofunctions = df['GO Functions']

    def __len__(self):
        return len(self.unitprotids)

    def __getitem__(self, idx):
        return (self.unitprotids[idx],self.residues[idx],self.gofunctions[idx])