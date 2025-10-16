from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
import lightning as pl
from torch.utils.data import DataLoader, random_split
import torch
from Model.PolyEncoder import CreativityScorer
from Model.Dataset import CreativityRankingDataset
from test import computeCorrelation
from lightning.pytorch.callbacks import ModelCheckpoint


def main():

    batch = 2
    epochs = 4
    devices = torch.cuda.device_count()
    pl.seed_everything(42)
    tokenizer = "bert-base-uncased"

    trainDataset = CreativityRankingDataset("/home/sam/datasets/train.csv", tokenizer)
    valDataset = CreativityRankingDataset("/home/sam/datasets/val.csv", tokenizer)

    # train_size = int(0.875 * len(dataset))
    # val_size = len(dataset) - train_size
    # train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(trainDataset, batch_size=batch, shuffle=True, num_workers=15)
    val_loader = DataLoader(valDataset, batch_size=batch, shuffle=False, num_workers=15)
    wandb_logger = WandbLogger(project="fixed-testing", name="rb-l")

    checkpoint_callback = ModelCheckpoint(
        monitor='val_pearson',
        mode='max',
        save_top_k=1,
        dirpath='home/sam/checkpoints/',
        filename='best-model'
    )

    model = CreativityScorer(tokenizer, wandb_logger)

    # freeze all encoder layers (including embedding)
    for param in model.model.encoder.parameters():
        param.requires_grad = False


    # layers unfrozen

    for layer in model.model.encoder.encoder.layer[-24:]:
        for param in layer.parameters():
            param.requires_grad = True

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=devices,
        precision="16",
        logger=wandb_logger,
        log_every_n_steps=10,
        accumulate_grad_batches=4,
        strategy=DDPStrategy(find_unused_parameters=True),
        gradient_clip_val=0.8,
        val_check_interval=0.20,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model, train_loader, val_loader)

    best_model = model

    testPath = "/home/sam/datasets/test.csv"

    correlation = computeCorrelation(best_model, testPath, batch, tokenizer, 128)

    wandb_logger.log_metrics({"correlation": correlation})

    heldOutPath = "/home/sam/datasets/HeldOutTest.csv"

    finalCorrelation = computeCorrelation(best_model, heldOutPath, batch, tokenizer, 128, ho=True)

    wandb_logger.log_metrics({"HeldOut correlation": finalCorrelation})

if __name__ == "__main__":
    main()
