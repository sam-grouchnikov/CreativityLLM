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
from lightning.pytorch.callbacks import ModelCheckpoint
from scipy.stats import pearsonr



import torch
import torch.nn.functional as F

from Model.PolyEncoder import CreativityScorer
from Model.Dataset import CreativityRankingDataset
from test import computeCorrelation


def main():

    batch = 2
    epochs = 10
    devices = torch.cuda.device_count()
    pl.seed_everything(42)
    tokenizer = "microsoft/deberta-v3-large"



    dataset = CreativityRankingDataset("/home/sam/datasets/TrainData.csv", tokenizer)
    train_size = int(0.875 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True, num_workers=15)
    val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False, num_workers=15)

    model = CreativityScorer(tokenizer)
    for param in model.model.encoder.parameters():
        param.requires_grad = False

    for layer in model.model.encoder.encoder.layer[-18:]:
        for param in layer.parameters():
            param.requires_grad = True

    wandb_logger = WandbLogger(project="poly-encoder-testing", name="deberta large ebs 8 no dl")


    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=devices,
        precision="16",
        logger=wandb_logger,
        log_every_n_steps=50,
        accumulate_grad_batches=4,
        strategy=DDPStrategy(find_unused_parameters=True),
    )
    trainer.fit(model, train_loader, val_loader)

    testPath = "/home/sam/datasets/TestData.csv"

    correlation = computeCorrelation(model, testPath, batch, tokenizer, 128)

    wandb_logger.log_metrics({"correlation": correlation})

    heldOutPath = "/home/sam/datasets/HeldOutTest.csv"

    finalCorrelation = computeCorrelation(model, heldOutPath, batch, tokenizer, 128, ho=True)

    wandb_logger.log_metrics({"HeldOut correlation": finalCorrelation})



if __name__ == "__main__":
    main()
