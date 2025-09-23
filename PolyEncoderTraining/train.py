from lightning.pytorch.callbacks import ModelCheckpoint

print("Starting imports")
import torch
from lightning import Trainer
from transformers import BertTokenizer
from pytorch_lightning.loggers import WandbLogger
from PolyEncoderTraining.PolyEncoder import PolyEncoder
from PolyEncoderTraining.CustomDataset import PairwiseDatasetLightning

batch_size = 128
epochs = 5
learning_rate = 3e-5
poly_m = 64
max_len = 64
devices = torch.cuda.device_count()

wandb_logger = WandbLogger(project="poly-encoder-iterations", name="test1")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer_name = "bert-base-uncased"


path = "/home/ubuntu/datasets/AllCut.csv"
dm = PairwiseDatasetLightning(path, tokenizer, batch_size=batch_size, max_len=max_len)

model = PolyEncoder(tokenizer_name, poly_m, learning_rate)

checkpoint_callback = ModelCheckpoint(
    dirpath="/home/ubuntu/checkpoints",
    filename="epoch-{epoch:02d}",
    save_top_k=-1,
    every_n_epochs=1,
    save_last=True
)

trainer = Trainer(
    max_epochs=epochs,
    accelerator="gpu",
    devices=devices,
    precision=16,
    logger=wandb_logger,
    log_every_n_steps=500,
    val_check_interval=1.0,
    callbacks=[checkpoint_callback],
    default_root_dir="/home/ubuntu",
    enable_checkpointing=True
)

trainer.fit(model, dm)

trainer.test(model, dm)
