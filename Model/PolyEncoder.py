import torch.nn as nn
from transformers import AutoModel
import lightning as pl
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr

from test import computeCorrelation


class PolyEncoder(nn.Module):
    def __init__(self, model_name, poly_m=256):
        super().__init__()
        self.model_name = model_name
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.poly_m = poly_m
        self.dropout = nn.Dropout(0.1)
        self.cross_attn = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(self.hidden_size)


        # Learnable poly codes
        self.poly_codes = nn.Embedding(poly_m, self.hidden_size)

        # Regression head for scoring
        self.reg_head = nn.Sequential(
            nn.Linear(self.hidden_size * 3, 1028),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1028, 1)
        )

        # Poly code indices for lookup
        self.register_buffer("poly_code_ids", torch.arange(poly_m))



    def encode_candidate(self, input_ids, attention_mask, token_type_ids=None):
        # Candidate encoder
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        cls_vec = outputs.last_hidden_state[:, 0, :]  # [B, H]
        return cls_vec

    def encode_context(self, input_ids, attention_mask, token_type_ids=None):
        # Context encoder
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        token_embeds = outputs.last_hidden_state

        # Poly codes
        poly_codes = self.poly_codes(self.poly_code_ids)  # [M, H]
        poly_codes = poly_codes.unsqueeze(0).expand(token_embeds.size(0), -1, -1)  # [B, M, H]

        # Attention: poly codes attend to tokens
        attn_scores = torch.matmul(poly_codes, token_embeds.transpose(1, 2))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attended = torch.bmm(attn_weights, token_embeds)
        return attended

    def forward(self, context_inputs, candidate_inputs):
        # Encode context and candidate
        context_vecs = self.encode_context(**context_inputs)  # [B, M, H]
        candidate_vec = self.encode_candidate(**candidate_inputs)  # [B, H]
        context_vecs = F.normalize(context_vecs, dim=-1)
        candidate_vec = F.normalize(candidate_vec, dim=-1)

        # Candidate attends to poly codes (single head attn variant)
        # attn_scores = torch.matmul(candidate_vec.unsqueeze(1), context_vecs.transpose(1, 2)).squeeze(1)  # [B, M]
        # attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, M]
        # context_pooled = torch.bmm(attn_weights.unsqueeze(1), context_vecs).squeeze(1)  # [B, H]

        # Q, K, V
        query = candidate_vec.unsqueeze(1)
        key = value = context_vecs

        # multi-head cross-attention
        attended, _ = self.cross_attn(query, key, value)
        context_pooled = self.norm(attended.squeeze(1))

        # Option 1: combine candidate and pooled context with dot product
        # score = torch.sum(context_pooled * candidate_vec, dim=-1, keepdim=True)  # [B, 1]

        # Option 2: regression head (maps to scalar if desired)
        combined = torch.cat((
            context_pooled,
            candidate_vec,
            candidate_vec,
        ), dim=1)

        score = self.reg_head(combined)

        return score.squeeze(-1)

    def getName(self):
        return self.model_name

class CreativityScorer(pl.LightningModule):
    def __init__(self, model_name, logger, poly_m=256, lr=1e-5):
        super().__init__()
        self.model_name = model_name
        self.model = PolyEncoder(model_name, poly_m)
        self.lr = lr
        self.val_pearson_ema = None
        self.ema_alpha = 0.5
        self.wandb_logger = logger

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
        preds = torch.cat(self.val_preds).numpy()
        labels = torch.cat(self.val_labels).numpy()

        pearson_corr, _ = pearsonr(preds, labels)
        if self.val_pearson_ema is None:
            self.val_pearson_ema = pearson_corr
        else:
            self.val_pearson_ema = (
                    self.ema_alpha * pearson_corr +
                    (1 - self.ema_alpha) * self.val_pearson_ema
            )
        self.log("val_pearson", pearson_corr, prog_bar=True)
        self.log("val_pearson_ema", self.val_pearson_ema, prog_bar=True)
        correlation = computeCorrelation(self, "/home/sam/datasets/test.csv", 16, "glaiveai/eelbert-tiny", 128)

        self.wandb_logger.log_metrics({"correlation": correlation})

        self.val_preds = []
        self.val_labels = []

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.encoder.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
                "lr": self.lr,
            },
            {
                "params": [p for n, p in self.model.encoder.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": self.lr,
            },
            {
                "params": [
                    *self.model.poly_codes.parameters(),
                    *self.model.reg_head.parameters(),
                    *self.model.cross_attn.parameters(),
                    *self.model.norm.parameters()
                ],
                "weight_decay": 0.01,
                "lr": self.lr / 5,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
        return optimizer