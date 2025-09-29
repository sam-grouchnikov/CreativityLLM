from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from scipy.stats import pearsonr
import pandas as pd
from scipy.special import expit

class CorrelationDataset(Dataset):
    def __init__(self, csv_file, tokenizer_name="bert-base-uncased", max_length=128):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        q_enc = self.tokenizer(
            row[0],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        r_enc = self.tokenizer(
            row[1],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "question_input": {k: v.squeeze(0) for k, v in q_enc.items()},
            "response_input": {k: v.squeeze(0) for k, v in r_enc.items()},
            "score": torch.tensor(row[2], dtype=torch.float)
        }

def computeCorrelation(model, csv_path, batch_size, tokenizer_name, max_length=128):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    dataset = CorrelationDataset(csv_path, tokenizer_name, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle = False)

    model.eval()
    preds = []
    targets = []

    device = "cuda"
    model.to(device)

    with torch.no_grad():
        for batch in dataloader:
            q_input = {k: v.to(device) for k, v in batch["question_input"].items()}
            r_input = {k: v.to(device) for k, v in batch["response_input"].items()}

            scores = model.model(q_input, r_input)  # PolyEncoder returns single score
            preds.append(scores.cpu())
            targets.append(batch["score"].cpu())

    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()

    preds_min = preds.min()
    preds_max = preds.max()
    print("Min, max: ", preds_min, preds_max)
    preds_norm = (preds - preds_min) / (preds_max - preds_min + 1e-8)

    # Print a few samples
    print("\nSample predictions (normalized) vs targets:")
    for i in range(min(10, len(preds_norm))):
        print(f"{i:2d}: pred_norm={preds_norm[i]:.4f} | target={targets[i]:.4f}")

    for i in range(min(20, len(preds))):  # only show first 10 to avoid spam
        print(f"Pred: {preds[i]:.4f} | Actual: {targets[i]:.4f}")

    pearson_corr = pearsonr(preds, targets)[0]
    print(f"Pearson correlation: {pearson_corr:.4f}")
    return pearson_corr