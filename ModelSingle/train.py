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

from ModelSingle.PolyEncoder import CreativityRanker2
from ModelSingle.CustomDataset import CreativityRankingDataset2
from test import computeCorrelation


def main():

    batch = 4
    epochs = 10
    devices = torch.cuda.device_count()
    pl.seed_everything(42)
    tokenizer = "microsoft/deberta-v3-large"



    dataset = CreativityRankingDataset2("/home/sam/datasets/Filtered.csv")
    train_size = int(0.875 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True, num_workers=15)
    val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False, num_workers=15)

    model = CreativityRanker2(tokenizer)
    for param in model.model.encoder.parameters():
        param.requires_grad = False

    for layer in model.model.encoder.encoder.layer[-12:]:
        for param in layer.parameters():
            param.requires_grad = True

    wandb_logger = WandbLogger(project="poly-encoder-testing", name="m=64, lr=3e-7, dbv3l, reglay, 12lay")

    # checkpoint_callback = ModelCheckpoint(
    #     monitor="val_loss",  # metric to monitor
    #     dirpath="/home/sam/checkpoints",  # directory to save checkpoints
    #     filename="best-checkpoint-{epoch:02d}-{val_loss:.2f}",  # filename template
    #     save_top_k=1,  # save only the best model
    #     mode="min",  # 'min' because lower val_loss is better
    #     save_weights_only=False  # set True to save only weights
    # )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=devices,
        # callbacks=[checkpoint_callback],
        precision="16",
        logger=wandb_logger,
        log_every_n_steps=5,
        accumulate_grad_batches=4,
        strategy=DDPStrategy(find_unused_parameters=True),
    )
    trainer.fit(model, train_loader, val_loader)

    testPath = "/home/sam/datasets/TestData.csv"

    correlation = computeCorrelation(model, testPath, batch, tokenizer, 128)

    wandb_logger.log_metrics({"correlation": correlation})



if __name__ == "__main__":
    main()
