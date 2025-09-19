import torch
from torch.utils.data import Dataset
import pandas as pd

class PairwiseDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file, names=["context", "candidate1", "score1", "candidate2", "score2"])

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        context = str(row["context"])
        cand1 = str(row["cand1"])
        cand2 = str(row["cand2"])
        score1 = torch.tensor(row["score1"], dtype=torch.float)
        score2 = torch.tensor(row["score2"], dtype=torch.float)

        return {
            "context": context,
            "candidate1": cand1,
            "candidate2": cand2,
            "score1": score1,
            "score2": score2,
        }