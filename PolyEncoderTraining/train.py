import torch
from lightning import Trainer
from pytorch_lightning import LightningModule
from transformers import BertTokenizer
from pytorch_lightning.loggers import WandbLogger

from CustomDataset import PairwiseDatasetLightning
from PolyEncoder import PolyEncoderLightning

batch_size = 128
epochs = 5
learning_rate = 3e-5
poly_m = 64
max_len = 64
devices = torch.cuda.device_count()

wandb_logger = WandbLogger(project="poly-encoder-iterations", name="test1")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer_name = "bert-base-uncased"

# Load full dataset
path = "/home/ubuntu/datasets/AllCut.csv"
dm = PairwiseDatasetLightning(path, tokenizer, batch_size=batch_size, max_len=max_len)

model: LightningModule = PolyEncoderLightning(tokenizer_name, poly_m)

trainer = Trainer(
    max_epochs=epochs,
    accelerator="gpu",
    devices = devices,
    precision=16,
    logger = wandb_logger,
    log_every_n_steps=500,
    default_root_dir="/home/ubuntu",
    enable_checkpointing=True,
)

trainer.fit(model, dm)

trainer.test(model, dm)
