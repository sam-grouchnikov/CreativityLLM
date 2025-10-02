from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, DebertaV2Tokenizer, DebertaV2TokenizerFast
from scipy.stats import pearsonr, spearmanr
import pandas as pd
from scipy.special import expit
import matplotlib.pyplot as plt


class CorrelationDataset(Dataset):
    def __init__(self, csv_file, tokenizer_name="microsoft/deberta-v3-base", max_length=128):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        question_text = row["prompt"]
        response_text = row["response"]

        q_enc = self.tokenizer(
            question_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        r_enc = self.tokenizer(
            response_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "question_input": {k: v.squeeze(0) for k, v in q_enc.items()},
            "response_input": {k: v.squeeze(0) for k, v in r_enc.items()},
            "score": torch.tensor(row["se"], dtype=torch.float),
            "question_text": question_text
        }

def computeCorrelation(model, csv_path, batch_size, tokenizer_name, max_length=128):
    dataset = CorrelationDataset(csv_path, tokenizer_name, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    preds = []
    targets = []
    questions = []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    with torch.no_grad():
        for batch in dataloader:
            q_input = {k: v.to(device) for k, v in batch["question_input"].items()}
            r_input = {k: v.to(device) for k, v in batch["response_input"].items()}

            scores = model.model(q_input, r_input)  # shape [batch_size]
            preds.append(scores.cpu())
            targets.append(batch["score"].cpu())
            questions.extend(batch["question_text"])  # raw text

    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()

    df = pd.DataFrame({
        "question": questions,
        "pred": preds,
        "target": targets,
    })


    pearson_corr = pearsonr(df["pred"], df["target"])[0]
    spearman_corr = spearmanr(df["pred"], df["target"])[0]

    print(f"Pearson correlation: {pearson_corr:.4f}")
    print(f"Spearman correlation: {spearman_corr:.4f}")

    df.to_csv("pred_vs_actual_perprompt.csv", index=False)

    # Plot with color per prompt
    # Factorize questions to numeric codes for coloring
    color_codes, uniques = pd.factorize(df["question"])

    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(df["target"], df["pred"], c=color_codes, cmap="tab20", alpha=0.6)
    plt.xlabel("Actual scores")
    plt.ylabel("Predicted scores (per-prompt normalized)")
    plt.title(f"Predicted vs Actual (r={pearson_corr:.2f}, rho={spearman_corr:.2f})")
    plt.grid(True)

    # Legend mapping
    handles = []
    labels = []
    for i, q in enumerate(uniques):
        handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=plt.cm.tab20(i % 20), markersize=8))
        labels.append(q)
    plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15),
               ncol=2, fontsize=8, title="Questions")

    plt.tight_layout()
    plt.savefig("pred_vs_actual_perprompt.png", dpi=150, bbox_inches="tight")
    plt.close()

    return pearson_corr