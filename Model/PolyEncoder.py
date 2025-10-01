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

class PolyEncoder(nn.Module):
    def __init__(self, model_name="bert-base-uncased", poly_m=64):
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
    def __init__(self, model_name="bert-base-uncased", poly_m=64, lr=4e-7):
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
        loss = F.margin_ranking_loss(s1, s2, label, margin = 0.07)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        s1, s2 = self.forward(batch)
        label = batch['label'].float()
        loss = F.margin_ranking_loss(s1, s2, label, margin = 0.07)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)