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

    ckpt_path = r"C:\Users\samgr\PycharmProjects\CreativityLLM\best_model.ckpt"

    print("Loading raw checkpoint state_dict...")
    ckpt = torch.load(ckpt_path, map_location="cpu")   # <-- SAFE ON PY 3.15

    print("Constructing model...")
    best_model = CreativityScorer(model_name=tokenizer, logger=None)

    print("Loading state_dict...")
    best_model.load_state_dict(ckpt["state_dict"], strict=False)

    print("Loaded model successfully.")

    testPath = r"C:\Users\samgr\PycharmProjects\CreativityLLM\TrainingData\TestData.csv"

    correlation = computeCorrelation(best_model, testPath, 64, tokenizer, 128)
    print("Correlation:", correlation)



if __name__ == "__main__":
    main()
