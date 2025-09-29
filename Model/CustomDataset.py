import torch
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel
import lightning as pl
import torch
import torch.nn.functional as F

class CreativityRankingDataset(Dataset):
    def __init__(self, csv_file, tokenizer_name="bert-base-uncased", max_length=128):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        question = str(row[0])
        r1 = str(row[1])
        r2 = str(row[2])

        q_inputs = self.tokenizer(question, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        r1_inputs = self.tokenizer(r1, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        r2_inputs = self.tokenizer(r2, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")

        row_label = float(row[3])
        label = 1 if row_label > 0 else -1

        return {
            "question_input": {k: v.squeeze(0) for k, v in q_inputs.items()},
            "response_1_input": {k: v.squeeze(0) for k, v in r1_inputs.items()},
            "response_2_input": {k: v.squeeze(0) for k, v in r2_inputs.items()},
            "label": torch.tensor(label, dtype=torch.long)
        }