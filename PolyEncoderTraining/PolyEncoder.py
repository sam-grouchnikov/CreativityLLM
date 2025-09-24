import torch
import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F
import lightning as L

def BCELoss(predict1, predict2, actual1, actual2, eps=1e-8):
    predictDiff = predict1 - predict2
    softTarget = actual1 / (actual1 + actual2 + eps)
    softTarget = softTarget.to(predict1.device).float()
    loss = -(softTarget * F.logsigmoid(predictDiff) +
             (1 - softTarget) * F.logsigmoid(-predictDiff))
    return loss.mean()

class PolyEncoder(L.LightningModule):
    def __init__(self, model_name, code_count, lr):
        super().__init__()
        self.save_hyperparameters()

        # BERT + poly-codes
        self.bert = BertModel.from_pretrained(model_name)
        self.lr = lr
        self.poly_codes = nn.Embedding(code_count, self.bert.config.hidden_size)

    def encodeContext(self, input_ids, attention_mask):
        """
            Encode a batch of context sequences into poly-encoder context vectors.

            Args:
                input_ids: [b, T]  -- token IDs of context sequences (batch of sequences)
                attention_mask: [b, T] -- mask indicating valid tokens (1) vs padding (0)

            Returns:
                context_vectors: [b, m, h]  -- poly-encoded context embeddings
                    b = batch size
                    m = number of poly codes
                    h = hidden size
        """

        # 1. Encode context sequences with BERT
        ctx_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # ctx_outputs: [b, T, h]

        # 2. Get poly codes and expand across batch
        poly_codes = self.poly_codes.weight.unsqueeze(0).expand(ctx_outputs.size(0), -1, -1)
        # poly_codes: [b, m, h]

        # 3. Compute raw attention scores: [b, m, T]
        attention = torch.bmm(poly_codes, ctx_outputs.transpose(1, 2))

        # 4. Mask out padding tokens before softmax
        # attention_mask: [b, T] -> [b, 1, T] to broadcast across m poly codes
        mask = attention_mask.unsqueeze(1)  # [b, 1, T]
        attention = attention.masked_fill(mask == 0, float('-inf'))

        # 5. Apply softmax over tokens
        attention_weights = torch.softmax(attention, dim=-1)

        # 6. Weighted sum over context tokens
        context_vectors = torch.bmm(attention_weights, ctx_outputs)
        # context_vectors: [b, m, h]

        return context_vectors

    def encodeCandidate(self, input_ids, attention_mask):
        """
                Encode candidate sequence by taking the [CLS] token embedding
                Args:
                    input_ids: [b, T]
                    attention_mask: [b, T]
                Returns:
                    cand_vec: [b, h]
        """

        # Encode candidate sequences with BERT and get a single vector per candidate
        cand_out = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # cand_out: [batch_size, seq_len, hidden_size]

        # Take the first token ([CLS] token) embedding as the candidate vector
        cand_vec = cand_out[:, 0, :]

        # cand_vec: [batch_size, hidden_size]
        return cand_vec

    def forward(self, ctx_input, ctx_mask, cand_input, cand_mask):
        """
        Compute relevance scores between context and multiple candidates.

        Args:
            ctx_input:  [b, T_ctx]
            ctx_mask:   [b, T_ctx]
            cand_input: [b, k, T_cand]  -- k candidates per context
            cand_mask:  [b, k, T_cand]

        Returns:
            scores: [b, k] -- similarity scores for each candidate
        """
        b, k, T_cand = cand_input.size()

        # Flatten candidates to run through BERT
        cand_input = cand_input.view(b * k, T_cand)
        cand_mask = cand_mask.view(b * k, T_cand)

        # Encode
        ctx_vecs = self.encodeContext(ctx_input, ctx_mask)  # [b, m, h]
        cand_vecs = self.encodeCandidate(cand_input, cand_mask)  # [b*k, h]
        cand_vecs = cand_vecs.view(b, k, -1)  # [b, k, h]

        # Compute attention between each context and candidate
        # Expand ctx_vecs to match candidates: [b, m, h] -> [b, k, m, h]
        ctx_vecs_exp = ctx_vecs.unsqueeze(1).expand(-1, k, -1, -1)  # [b, k, m, h]
        cand_vecs_exp = cand_vecs.unsqueeze(2)  # [b, k, 1, h]

        attn_scores = torch.matmul(ctx_vecs_exp, cand_vecs_exp.transpose(-1, -2)).squeeze(-1)  # [b, k, m]
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [b, k, m]

        ctx_final = torch.matmul(attn_weights.unsqueeze(2), ctx_vecs_exp).squeeze(2)  # [b, k, h]

        scores = torch.sum(ctx_final * cand_vecs, dim=-1)  # [b, k]
        return scores

    def training_step(self, batch, batch_idx):
        ctx_input, ctx_mask, cand_inputs, cand_masks, scores = batch
        # ctx_input: [B, T]
        # cand_inputs: [B, 2, T]
        # scores: [B, 2]

        preds = self(ctx_input, ctx_mask, cand_inputs, cand_masks)  # [B, 2]

        loss = BCELoss(preds[:, 0], preds[:, 1], scores[:, 0], scores[:, 1])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        ctx_input, ctx_mask, cand_inputs, cand_masks, scores = batch
        preds = self(ctx_input, ctx_mask, cand_inputs, cand_masks)  # [B, 2]

        loss = BCELoss(preds[:, 0], preds[:, 1], scores[:, 0], scores[:, 1])
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)