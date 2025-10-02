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

class CreativityRankingDataset2(Dataset):
    def __init__(self, csv_file, tokenizer_name="bert-base-uncased", max_length=128):
        self.data = pd.read_csv(csv_file, header=0)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        question = str(row["prompt"])
        r = str(row["response"])

        q_inputs = self.tokenizer(question, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        r_inputs = self.tokenizer(r, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")

        row_label = float(row["se"])

        return {
            "question_input": {k: v.squeeze(0) for k, v in q_inputs.items()},
            "response_input": {k: v.squeeze(0) for k, v in r_inputs.items()},
            "label": torch.tensor(row_label, dtype=torch.long)
        }