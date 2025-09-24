import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import lightning as pl

class ScoreDataModule(pl.LightningDataModule):
    def __init__(self, csv_path, tokenizer, batch_size=32, max_len=128, seed=42):
        super().__init__()
        self.csv_path = csv_path
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_len = max_len
        self.seed = seed

    class ScoreDataset(Dataset):
        def __init__(self, data, tokenizer, max_len):
            self.data = data
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            row = self.data.iloc[idx]

            def encode(text):
                enc = self.tokenizer(
                    str(text),
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_len,
                    return_tensors="pt"
                )
                return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)

            ctx_input, ctx_mask = encode(row.iloc[5])  # column 6
            cand_input, cand_mask = encode(row.iloc[6])  # column 7
            score = torch.tensor(float(row.iloc[8]), dtype=torch.float)  # column 9

            return ctx_input, ctx_mask, cand_input, cand_mask, score

    def setup(self, stage=None):
        # Read CSV
        data = pd.read_csv(self.csv_path, header=0)
        dataset = self.ScoreDataset(data, self.tokenizer, self.max_len)

        total_size = len(dataset)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size

        generator = torch.Generator().manual_seed(self.seed)

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [train_size, val_size, test_size], generator=generator
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True
        )
