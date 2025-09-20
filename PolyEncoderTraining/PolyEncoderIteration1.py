import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from transformers import BertModel, BertTokenizer, DistilBertTokenizer
import wandb

from PolyEncoderTraining.CustomDataset import PairwiseDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

wandb.init(project='poly-encoder-iterations', name='poly-encoder-iteration-1')

# Hyperparameters
batch_size = 64
epochs = 3
learning_rate = 3e-5
poly_m = 64

class PolyEncoder(nn.Module):
    def __init__(self, model_name = "bert-base-uncased", code_count = poly_m):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.poly_codes = nn.Embedding(code_count, self.bert.config.hidden_size)
        self.code_count = code_count

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
                Compute relevance score between context and candidate.
                Returns:
                    scores: [b]  -- max similarity across poly codes
        """

        ctx_vecs = self.encodeContext(ctx_input, ctx_mask) # [b, m, h]
        cand_vecs = self.encodeCandidate(cand_input, cand_mask) # [b, h]

        # [b, m, h] dot [b, h] -> [b, m]
        attn_scores = torch.bmm(ctx_vecs, cand_vecs.unsqueeze(-1)).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [b, m]

        # Weighted sum of poly codes
        ctx_final = torch.bmm(attn_weights.unsqueeze(1), ctx_vecs).squeeze(1)  # [b, h]

        # Final similarity score
        scores = torch.sum(ctx_final * cand_vecs, dim=-1)  # [b]

        return scores

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = PolyEncoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def BCELoss(predict1, predict2, actual1, actual2, eps=1e-8):
    predictDiff = predict1 - predict2
    softTarget = actual1 / (actual1 + actual2 + eps)
    softTarget = softTarget.to(predict1.device).float()
    loss = -(softTarget * F.logsigmoid(predictDiff) +
             (1 - softTarget) * F.logsigmoid(-predictDiff))
    return loss.mean()

# Load full dataset
path = "C:\\Users\\samgr\\PycharmProjects\\CreativityLLM\\TrainingData\\PairwiseComparisons\\MindReading\\MindReadingPairsCut.csv"
full_dataset = PairwiseDataset(path, tokenizer)

# Shuffle and compute split lengths
total_len = len(full_dataset)
train_len = int(0.7 * total_len)
val_len   = int(0.1 * total_len)
test_len  = total_len - train_len - val_len

# Use random_split to split dataset
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_len, val_len, test_len])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

global_step = 0

for epoch in range(epochs):
    model.train()
    train_epoch_loss = 0.0

    for batch in train_loader:
        global_step += 1
        optimizer.zero_grad()

        ctx_input = batch['ctx_input'].to(device)
        ctx_mask = batch['ctx_mask'].to(device)

        # Candidate 1
        cand1_input = batch["cand1_input"].to(device)
        cand1_mask = batch["cand1_mask"].to(device)
        score1 = batch["score1"].to(device)

        # Candidate 2
        cand2_input = batch["cand2_input"].to(device)
        cand2_mask = batch["cand2_mask"].to(device)
        score2 = batch["score2"].to(device)

        # Forward pass
        pred1 = model(ctx_input, ctx_mask, cand1_input, cand1_mask)
        pred2 = model(ctx_input, ctx_mask, cand2_input, cand2_mask)

        # Loss
        loss = BCELoss(pred1, pred2, score1, score2)
        loss.backward()
        optimizer.step()

        train_epoch_loss += loss.item()

        # Step-level logging
        wandb.log({"train_step_loss": loss.item()}, step=global_step)

    # Average training loss for the epoch
    train_epoch_loss /= len(train_loader)
    wandb.log({"train_epoch_loss": train_epoch_loss}, step=global_step)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_batch in val_loader:
            pred1 = model(val_batch['ctx_input'].to(device), val_batch['ctx_mask'].to(device),
                          val_batch['cand1_input'].to(device), val_batch['cand1_mask'].to(device))
            pred2 = model(val_batch['ctx_input'].to(device), val_batch['ctx_mask'].to(device),
                          val_batch['cand2_input'].to(device), val_batch['cand2_mask'].to(device))
            loss = BCELoss(pred1, pred2, val_batch['score1'].to(device), val_batch['score2'].to(device))
            val_loss += loss.item()
    val_loss /= len(val_loader)
    print(f"Epoch {epoch}, train loss: {train_epoch_loss:.4f}, validation loss: {val_loss:.4f}")

    # Epoch-level validation logging
    wandb.log({"val_loss": val_loss}, step=global_step)