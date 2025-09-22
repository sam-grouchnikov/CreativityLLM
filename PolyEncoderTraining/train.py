import torch
from lightning import Trainer
from transformers import BertTokenizer
from pytorch_lightning.loggers import WandbLogger

from PolyEncoderTraining.CustomDataset import PairwiseDatasetLightning
from PolyEncoderTraining.PolyEncoder import PolyEncoderLightning

batch_size = 128
epochs = 3
learning_rate = 3e-5
poly_m = 64
max_len = 128
devices = torch.cuda.device_count()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

wandb_logger = WandbLogger(project="poly-encoder-iterations", name="test1")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer_name = "bert-base-uncased"

# Load full dataset
path = "/home/ubuntu/datasets/AllCut.csv"
dm = PairwiseDatasetLightning(path, tokenizer, batch_size=batch_size, max_len=max_len)

model = PolyEncoderLightning(tokenizer_name, poly_m, learning_rate)

trainer = Trainer(
    max_epochs=epochs,
    accelerator="gpu",
    devices = devices,
    logger = wandb_logger,
    log_every_n_steps=500
)

trainer.fit(model, dm)

trainer.test(model, dm)
