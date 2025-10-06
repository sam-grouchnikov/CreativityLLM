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
from scipy.stats import pearsonr, spearmanr

from test import computeCorrelation


class PolyEncoder(nn.Module):
    def __init__(self, model_name, poly_m=64):
        super().__init__()
        self.model_name = model_name
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.poly_m = poly_m

        # Learnable poly codes
        self.poly_codes = nn.Embedding(poly_m, self.hidden_size)

        # Regression head for scoring
        self.reg_head = nn.Linear(self.hidden_size, 1)

        # Poly code indices
        self.register_buffer("poly_code_ids", torch.arange(poly_m))

    def encode_context(self, input_ids, attention_mask, token_type_ids=None):

        # Encodes context into M poly-code embeddings
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        token_embeds = outputs.last_hidden_state  # [B, L, H]

        # Poly codes
        poly_codes = self.poly_codes(self.poly_code_ids)  # [M, H]
        poly_codes = poly_codes.unsqueeze(0).expand(token_embeds.size(0), -1, -1)  # [B, M, H]

        # Attention: poly codes attend to tokens
        attn_scores = torch.matmul(poly_codes, token_embeds.transpose(1, 2))  # [B, M, L]
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, M, L]
        attended = torch.bmm(attn_weights, token_embeds)  # [B, M, H]
        return attended  # [B, M, H]

    def encode_candidate(self, input_ids, attention_mask, token_type_ids=None):

        # Encodes candidate into a single vector
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        cls_vec = outputs.last_hidden_state[:, 0, :]  # [B, H]
        return cls_vec

    def forward(self, context_inputs, candidate_inputs):
        # Encode context and candidate
        context_vecs = self.encode_context(**context_inputs)  # [B, M, H]
        candidate_vec = self.encode_candidate(**candidate_inputs)  # [B, H]

        # Candidate attends to poly codes
        attn_scores = torch.matmul(candidate_vec.unsqueeze(1), context_vecs.transpose(1, 2)).squeeze(1)  # [B, M]
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, M]
        context_pooled = torch.bmm(attn_weights.unsqueeze(1), context_vecs).squeeze(1)  # [B, H]

        # Option 1: combine candidate and pooled context with dot product
        # score = torch.sum(context_pooled * candidate_vec, dim=-1, keepdim=True)  # [B, 1]

        # Option 2: regression head (maps to scalar if desired)
        score = self.reg_head(context_pooled)  # [B, 1]

        return score.squeeze(-1)  # [B]

    def getName(self):
        return self.model_name

class CreativityScorer(pl.LightningModule):
    def __init__(self, model_name, poly_m=64, lr=3e-5):
        super().__init__()
        self.model_name = model_name
        self.model = PolyEncoder(model_name, poly_m)
        self.lr = lr

        # Validation train metric tracking
        self.val_preds = []
        self.val_labels = []

    def forward(self, batch):
        pred = self.model(batch['question_input'], batch['response_input'])
        return pred

    def training_step(self, batch, batch_idx):
        pred = self(batch)
        label = batch['label'].float()
        # Use smooth L1 loss, combination of MAE and MSE
        loss = F.smooth_l1_loss(pred, label)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self(batch)
        label = batch['label'].float()
        loss = F.smooth_l1_loss(pred, label)
        self.log("val_loss", loss, prog_bar=True)

        # Track correlations
        self.val_preds.append(pred.detach().cpu())
        self.val_labels.append(label.detach().cpu())
        return loss

    def on_validation_epoch_end(self):
        # Logging correlations after each epoch
        preds = torch.cat(self.val_preds).numpy()
        labels = torch.cat(self.val_labels).numpy()

        pearson_corr, _ = pearsonr(preds, labels)
        self.log("val_pearson", pearson_corr, prog_bar=True)

        self.val_preds = []
        self.val_labels = []

        testCorr = computeCorrelation(self, "/home/sam/datasets/TestData.csv", self.model_name, 2, 128)
        self.log("test_corr", testCorr, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

