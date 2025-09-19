import torch
from torch.utils.data import Dataset
import pandas as pd

class PairwiseDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len=128):
        self.data = pd.read_csv(csv_path, names=["context", "candidate1", "score1", "candidate2", "score2"])
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        context = str(row["context"])
        cand1   = str(row["candidate1"])
        cand2   = str(row["candidate2"])
        score1  = torch.tensor(row["score1"], dtype=torch.float)
        score2  = torch.tensor(row["score2"], dtype=torch.float)

        ctx_enc = self.tokenizer(context, padding="max_length", truncation=True,
                                 max_length=self.max_len, return_tensors="pt")

        cand1_enc = self.tokenizer(cand1, padding="max_length", truncation=True,
                                   max_length=self.max_len, return_tensors="pt")
        cand2_enc = self.tokenizer(cand2, padding="max_length", truncation=True,
                                   max_length=self.max_len, return_tensors="pt")

        return {
            "ctx_input": ctx_enc["input_ids"].squeeze(0),
            "ctx_mask": ctx_enc["attention_mask"].squeeze(0),

            "cand1_input": cand1_enc["input_ids"].squeeze(0),
            "cand1_mask": cand1_enc["attention_mask"].squeeze(0),
            "score1": score1,

            "cand2_input": cand2_enc["input_ids"].squeeze(0),
            "cand2_mask": cand2_enc["attention_mask"].squeeze(0),
            "score2": score2
        }