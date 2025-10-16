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

    tokenizer = "bert-base-uncased"

    best_model = CreativityScorer.load_from_checkpoint("/home/sam/checkpoints/best_model.ckpt")

    testPath = "/home/sam/datasets/test.csv"

    correlation = computeCorrelation(best_model, testPath, 64, tokenizer, 128)





if __name__ == "__main__":
    main()
