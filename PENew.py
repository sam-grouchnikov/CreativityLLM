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
import torch
import torch.nn.functional as F

class CreativityRankingDataset(Dataset):
    def __init__(self, csv_file, tokenizer_name="bert-base-uncased", max_length=128):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def _safe_str(self, x):
        return x if isinstance(x, str) else ""

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        question = self._safe_str(row[0])
        r1, s1 = self._safe_str(row[1]), row[2]
        r2, s2 = self._safe_str(row[3]), row[4]

        q_inputs = self.tokenizer(question, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        r1_inputs = self.tokenizer(r1, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        r2_inputs = self.tokenizer(r2, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")

        label = 1 if s1 > s2 else -1  # Margin ranking label

        return {
            "question_input": {k: v.squeeze(0) for k, v in q_inputs.items()},
            "response_1_input": {k: v.squeeze(0) for k, v in r1_inputs.items()},
            "response_2_input": {k: v.squeeze(0) for k, v in r2_inputs.items()},
            "label": torch.tensor(label, dtype=torch.long)
        }


class PolyEncoder(nn.Module):
    def __init__(self, model_name="bert-base-uncased", poly_m=16):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.poly_m = poly_m

        # Learnable poly codes
        self.poly_codes = nn.Embedding(poly_m, self.hidden_size)

        # Init poly-code indices
        self.register_buffer("poly_code_ids", torch.arange(poly_m))

    def encode_question(self, input_ids, attention_mask, token_type_ids=None):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        token_embeddings = output.last_hidden_state  # [B, L, H]

        poly_codes = self.poly_codes(self.poly_code_ids)  # [M, H]
        poly_codes = poly_codes.unsqueeze(0).expand(token_embeddings.size(0), -1, -1)  # [B, M, H]

        # Attention: poly codes attend to question tokens
        attn_scores = torch.matmul(poly_codes, token_embeddings.transpose(1, 2))  # [B, M, L]
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, M, L]
        attended = torch.bmm(attn_weights, token_embeddings)  # [B, M, H]
        return attended  # [B, M, H]

    def encode_response(self, input_ids, attention_mask, token_type_ids=None):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return output.last_hidden_state[:, 0, :]  # CLS token: [B, H]

    def forward(self, question_inputs, response_inputs):
        # Encode question and responses
        q_vecs = self.encode_question(**question_inputs)  # [B, M, H]
        r_vec = self.encode_response(**response_inputs)   # [B, H]

        # Attention: response attends to poly codes
        attn_scores = torch.matmul(r_vec.unsqueeze(1), q_vecs.transpose(1, 2)).squeeze(1)  # [B, M]
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, M]
        context_vec = torch.bmm(attn_weights.unsqueeze(1), q_vecs).squeeze(1)  # [B, H]

        # Dot-product similarity
        score = torch.sum(context_vec * r_vec, dim=-1)  # [B]
        return score

class CreativityRanker(pl.LightningModule):
    def __init__(self, model_name="bert-base-uncased", poly_m=16, lr=2e-5):
        super().__init__()
        self.model = PolyEncoder(model_name, poly_m)
        self.lr = lr

    def forward(self, batch):
        q_input = batch['question_input']
        r1_input = batch['response_1_input']
        r2_input = batch['response_2_input']

        s1 = self.model(q_input, r1_input)
        s2 = self.model(q_input, r2_input)
        return s1, s2

    def training_step(self, batch, batch_idx):
        s1, s2 = self.forward(batch)
        label = batch['label'].float()
        loss = F.margin_ranking_loss(s1, s2, label, margin=0.2)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

def main():
    dataset = CreativityRankingDataset("/home/sam/datasets/AllCut.csv")
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    model = CreativityRanker()
    wandb_logger = WandbLogger(project="poly-encoder-iterations", name="test1")

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="gpu",
        devices=3,
        precision="16",
        logger=wandb_logger,
        log_every_n_steps=1,
        limit_val_batches=0,
        strategy=DDPStrategy(find_unused_parameters=True)
    )
    trainer.fit(model, dataloader)

if __name__ == "__main__":
    main()
