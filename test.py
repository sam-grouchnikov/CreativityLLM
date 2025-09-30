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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    preds = []
    targets = []
    questions = []  # keep raw question text

    device = "cuda"
    model.to(device)

    with torch.no_grad():
        for batch in dataloader:
            q_input = {k: v.to(device) for k, v in batch["question_input"].items()}
            r_input = {k: v.to(device) for k, v in batch["response_input"].items()}

            scores = model.model(q_input, r_input)  # PolyEncoder returns single score
            preds.append(scores.cpu())
            targets.append(batch["score"].cpu())
            questions.extend(batch["question_text"])  # <--- raw question text

    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()

    # Put everything in a DataFrame for easier per-prompt operations
    df = pd.DataFrame({
        "question": questions,
        "pred": preds,
        "target": targets,
    })

    # Normalize predictions within each question group
    def normalize_group(x):
        pmin, pmax = x.min(), x.max()
        if abs(pmax - pmin) < 1e-8:
            return x * 0  # flat predictions, normalize to 0
        return (x - pmin) / (pmax - pmin)

    df["pred_norm"] = df.groupby("question")["pred"].transform(normalize_group)

    # Compute correlations using normalized scores
    pearson_corr = pearsonr(df["pred_norm"], df["target"])[0]
    spearman_corr = spearmanr(df["pred_norm"], df["target"])[0]

    print(f"Pearson correlation (per-prompt normalized): {pearson_corr:.4f}")
    print(f"Spearman correlation (per-prompt normalized): {spearman_corr:.4f}")

    # Save CSV for inspection
    df.to_csv("pred_vs_actual_perprompt.csv", index=False)

    # --- Plotting ---
    plt.figure(figsize=(7, 7))
    # assign each unique question a color
    colors = pd.factorize(df["question"])[0]
    plt.scatter(df["target"], df["pred_norm"], c=colors, cmap="tab20", alpha=0.6)
    plt.xlabel("Actual scores")
    plt.ylabel("Predicted scores (per-prompt normalized)")
    plt.title(f"Predicted vs Actual (r={pearson_corr:.2f}, rho={spearman_corr:.2f})")
    plt.grid(True)

    # Save instead of show (since SSH won't display)
    plt.savefig("pred_vs_actual_perprompt.png", dpi=150, bbox_inches="tight")
    plt.close()

    return pearson_corr