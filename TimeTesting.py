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

    tokenizer = "microsoft/deberta-v3-large"

    print("Starting load")

    best_model = CreativityScorer.load_from_checkpoint("C:\\Users\\samgr\\PycharmProjects\\CreativityLLM\\best_model.ckpt", model_name=tokenizer, logger=None)

    print("Loaded")

    testPath = "C:\\Users\\samgr\\PycharmProjects\\CreativityLLM\\TrainingData\\TestData.csv"

    correlation = computeCorrelation(best_model, testPath, 64, tokenizer, 128)





if __name__ == "__main__":
    main()
