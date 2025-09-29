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
from torch.utils.data import DataLoader, random_split

import torch
import torch.nn.functional as F

from Model.CustomDataset import CreativityRankingDataset
from Model.PolyEncoder import CreativityRanker


def main():

    batch = 128
    epochs = 1
    devices = 3
    pl.seed_everything(42)



    dataset = CreativityRankingDataset("/home/sam/datasets/TrainData.csv")
    train_size = int(0.875 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True, num_workers=15)
    val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False, num_workers=15)

    model = CreativityRanker()
    for param in model.model.encoder.parameters():
        param.requires_grad = False

    for layer in model.model.encoder.encoder.layer[-2:]:
        for param in layer.parameters():
            param.requires_grad = True

    wandb_logger = WandbLogger(project="poly-encoder-iterations", name="test1")

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=devices,
        precision="16",
        logger=wandb_logger,
        log_every_n_steps=1,
        strategy=DDPStrategy(find_unused_parameters=True)
    )
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
