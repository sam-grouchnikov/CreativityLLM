import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizer
import wandb
import time

from PolyEncoderTraining.CustomDataset import PairwiseDataset
from PolyEncoderTraining.PolyEncoder import PolyEncoder

batch_size = 64
epochs = 3
learning_rate = 3e-5
poly_m = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

wandb.init(project='poly-encoder-iterations', name='poly-encoder-iteration-1')

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
total_batches = len(train_loader)

for epoch in range(epochs):
    model.train()
    train_epoch_loss = 0.0
    epoch_start_time = time.time()

    for batch_idx, batch in enumerate(train_loader, start=1):
        batch_start_time = time.time()
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

        # Step-level logging to W&B
        wandb.log({"train_step_loss": loss.item()}, step=global_step)

        # Batch timing
        batch_time = time.time() - batch_start_time
        elapsed_time = time.time() - epoch_start_time
        batches_left = total_batches - batch_idx
        est_total_time = elapsed_time / batch_idx * total_batches
        est_remaining = est_total_time - elapsed_time

        print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{total_batches}, "
              f"Step Loss: {loss.item():.4f}, "
              f"Batch Time: {batch_time:.2f}s, "
              f"Elapsed: {elapsed_time:.2f}s, "
              f"ETA: {est_remaining:.2f}s")

    # Average training loss for the epoch
    train_epoch_loss /= total_batches
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

    print(f"Epoch {epoch+1} completed. Train Loss: {train_epoch_loss:.4f}, Validation Loss: {val_loss:.4f}")
    wandb.log({"val_loss": val_loss}, step=global_step)