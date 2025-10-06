from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, DebertaV2Tokenizer, DebertaV2TokenizerFast
from scipy.stats import pearsonr, spearmanr
import pandas as pd
import os
from scipy.special import expit
import matplotlib.pyplot as plt


class CorrelationDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=128, cache_path=None):
        self.max_length = max_length
        self.tokenizer = tokenizer

        if cache_path and os.path.exists(cache_path):
            self.encodings = torch.load(cache_path)
        else:
            df = pd.read_csv(csv_file)
            self.questions = df["prompt"].tolist()
            self.responses = df["response"].tolist()
            self.scores = torch.tensor(df["score"].values, dtype=torch.float)

            q_enc = self.tokenizer(
                self.questions,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
            r_enc = self.tokenizer(
                self.responses,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )

            self.encodings = {
                "question_input": q_enc,
                "response_input": r_enc,
                "score": self.scores,
                "question_text": self.questions,
            }

            if cache_path:
                torch.save(self.encodings, cache_path)
                print(f"💾 Saved tokenized cache to {cache_path}")

    def __len__(self):
        return len(self.encodings["score"])

    def __getitem__(self, idx):
        return {
            "question_input": {k: v[idx] for k, v in self.encodings["question_input"].items()},
            "response_input": {k: v[idx] for k, v in self.encodings["response_input"].items()},
            "score": self.encodings["score"][idx],
            "question_text": self.encodings["question_text"][idx],
        }


def computeCorrelation(model, csv_path, batch_size, tokenizer_name, max_length=128,
                       cache_path=None, ho=False):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    dataset = CorrelationDataset(csv_path, tokenizer, max_length=max_length, cache_path=cache_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    preds, targets, questions = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            q_input = {k: v.to(device, non_blocking=True) for k, v in batch["question_input"].items()}
            r_input = {k: v.to(device, non_blocking=True) for k, v in batch["response_input"].items()}
            score_true = batch["score"].to(device, non_blocking=True)

            score_pred = model.model(q_input, r_input)

            preds.append(score_pred.cpu())
            targets.append(score_true.cpu())
            questions.extend(batch["question_text"])

    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()

    df = pd.DataFrame({
        "question": questions,
        "pred": preds,
        "target": targets,
    })

    pearson_corr = pearsonr(df["pred"], df["target"])[0]

    if not ho:
        df.to_csv("pred_vs_actual.csv", index=False)

    # spearman_corr = spearmanr(df["pred"], df["target"])[0]

    # print(f"Pearson correlation: {pearson_corr:.4f}")
    # print(f"Spearman correlation: {spearman_corr:.4f}")



    # plt.figure(figsize=(10, 18))
    # plt.xlabel("Actual scores")
    # plt.ylabel("Predicted scores (per-prompt normalized)")
    # plt.title(f"Predicted vs Actual (r={pearson_corr:.2f}, rho={spearman_corr:.2f})")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig("pred_vs_actual_perprompt.png", dpi=150, bbox_inches="tight")
    # plt.close()
    #
    # if (not ho):
    #     df.to_csv("pred_vs_actual_perprompt.csv", index=False)
    #     plt.savefig("pred_vs_actual_perprompt.png", dpi=150, bbox_inches="tight")
    #
    # else:
    #     df.to_csv("pred_vs_actual_perprompt_ho.csv", index=False)
    #     plt.savefig("pred_vs_actual_perprompt_ho.png", dpi=150, bbox_inches="tight")

    return pearson_corr