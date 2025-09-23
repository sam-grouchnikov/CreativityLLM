import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import pytorch_lightning as pl

class PairwiseDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len):
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

class PairwiseDatasetLightning(pl.LightningDataModule):
    def __init__(self, data_csv, tokenizer, batch_size, max_len, seed = 42):
        super().__init__()
        self.data_csv = data_csv
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_len = max_len
        self.seed = seed

    def setup(self, stage=None):
        dataset = PairwiseDataset(self.data_csv, self.tokenizer, self.max_len)

        total_size = len(dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size

        generator = torch.Generator().manual_seed(self.seed)

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=generator)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)