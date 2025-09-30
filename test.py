from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from scipy.stats import pearsonr, spearmanr
import pandas as pd
from scipy.special import expit
import matplotlib.pyplot as plt


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

    pearson_corr = pearsonr(preds, targets)[0]
    spearman_corr = spearmanr(preds, targets)[0]
    print(f"Pearson correlation: {pearson_corr:.4f}")
    print(f"Spearman correlation: {spearman_corr:.4f}")

    plt.figure(figsize=(6, 6))
    plt.scatter(targets, preds_norm, alpha=0.6)
    plt.xlabel("Actual scores")
    plt.ylabel("Predicted scores (normalized)")
    plt.title(f"Predicted vs Actual (r={pearson_corr:.2f}, rho={spearman_corr:.2f})")
    plt.grid(True)
    plt.show()
    plt.savefig("pred_vs_actual.png")
    return pearson_corr